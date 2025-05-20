using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using HtmlAgilityPack;
using OpenAI;
using OpenAI.Chat;
using OpenAI.Embeddings;
using RAGBaseApp.Models;

class Program
{


    static async Task Main(string[] args)
    {

     // await PopulateEmbeddingsDB();

       await QueryBGBBot();
    }

    private async static Task QueryBGBBot()
    {
        string embeddingPath = GetProjectPath("Data/bgb_embeddings_new.json");
        List<Chunk> allChunks;

        // ✅ Load or generate embeddings
        if (File.Exists(embeddingPath))
        {
            allChunks = new List<Chunk>();

            Console.WriteLine("📂 Streaming embeddings from disk...");

            using FileStream fs = File.OpenRead(embeddingPath);

            await foreach (var chunk in JsonSerializer.DeserializeAsyncEnumerable<Chunk>(fs, new JsonSerializerOptions
            {
                AllowTrailingCommas = true,
                PropertyNameCaseInsensitive = true
            }))
            {
                if (chunk != null)
                {
                    allChunks.Add(chunk);
                }
            }

            // ✅ Get user question
            Console.Write("❓ Enter your question: ");
            string question = Console.ReadLine();

            string apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
            if (string.IsNullOrWhiteSpace(apiKey))
            {
                Console.WriteLine("❌ OPENAI_API_KEY environment variable not found.");
                return;
            }

            bool isLegalQuery = question.Contains("§") || question.ToLower().Contains("bgb");
            ChatClient chatClient = new(model: "gpt-4o", apiKey: apiKey);

            if (isLegalQuery)
            {
                Console.WriteLine("⚖️ BGB detected. Using local RAG...");

                EmbeddingClient queryEmbedClient = new("text-embedding-3-small", apiKey);
                var queryEmbedding = await queryEmbedClient.GenerateEmbeddingAsync(question);
                var queryVector = queryEmbedding.Value.ToFloats().ToArray();

                var topChunks = allChunks
                    .Select(c => new { Chunk = c, Score = CosineSimilarity(queryVector, c.Embedding) })
                    .OrderByDescending(x => x.Score)
                    .Take(3)
                    .ToList();

                string prompt = "You are a legal assistant. Use only the content below from the BGB.\r\n\r\nWhen citing legal references, refer to the **§ number(s) mentioned in the text**, not the internal section IDs (e.g., p0001, p2649).\r\n\r\nClearly explain tenant rights using plain, helpful language. Where applicable, mention the § number (e.g., § 535 BGB) to support your explanation.";
                foreach (var c in topChunks)
                {
                    prompt += $"[Section: {c.Chunk.Id}] {c.Chunk.Text}\n\n";
                }
                prompt += $"---\nQuestion: {question}";

                var response = await chatClient.CompleteChatAsync(prompt);
                Console.WriteLine($"\n📘 GPT (BGB grounded):\n{response.Value.Content[0].Text}");
            }
            else
            {
                Console.WriteLine("💬 No BGB trigger. Sending directly to GPT...");

                var response = await chatClient.CompleteChatAsync(question);
                Console.WriteLine($"\n🧠 GPT says:\n{response.Value.Content[0].Text}");
            }
        }
        else
        {
            Console.WriteLine("No embeddings found - exiting process");
            return;
        }
    }

    private async static Task PopulateEmbeddingsDB()
    {
        string apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            Console.WriteLine("OPENAI_API_KEY environment variable not found.");
            return;
        }

        EmbeddingClient client = new("text-embedding-3-small", apiKey);

        // Load HTML
        HtmlDocument doc = new HtmlDocument();
        string htmlPath = GetProjectPath("Data/bgb.html");
        doc.Load(htmlPath);


        //  Select paragraphs that contain <a name="..."> but not <br> (skip section headers)
        var nodes = doc.DocumentNode.SelectNodes("//p[a[@name] and not(br) and string-length(normalize-space()) > 10]");
        if (nodes == null || nodes.Count == 0)
        {
            Console.WriteLine("No matching <p> nodes found.");
            return;
        }

        Console.WriteLine($"🔍 Found {nodes.Count} legal chunks.");

        List<Chunk> allChunks = new();
        List<string> textsToEmbed = new();

        foreach (var node in nodes)
        {
            string id = node.SelectSingleNode(".//a")?.GetAttributeValue("name", null);

            if (string.IsNullOrWhiteSpace(id)) continue;

            string text = HtmlEntity.DeEntitize(node.InnerText)
                .Replace("\r", " ")
                .Replace("\n", " ")
                .Trim();

            while (text.Contains("  ")) text = text.Replace("  ", " ");

            if (!IsValidText(text))
            {
                Console.WriteLine($"⚠️ Skipping chunk {id} (invalid text, len = {text.Length})");
                continue;
            }

            allChunks.Add(new Chunk
            {
                Id = id,
                Text = text,
                Embedding = null  // placeholder, will be filled in next step
            });

            textsToEmbed.Add(text);
        }

        try
        {
            // ✅ Filter both lists in sync
            List<Chunk> filteredChunks = new();
            List<string> filteredTexts = new();

            for (int i = 0; i < allChunks.Count; i++)
            {
                var text = allChunks[i].Text;

                if (!string.IsNullOrWhiteSpace(text) && text.Length >= 10 && text.Length <= 10000)
                {
                    filteredChunks.Add(allChunks[i]);
                    filteredTexts.Add(text.Trim());
                }
                else
                {
                    Console.WriteLine($"⚠️ Filtering out invalid text at index {i} (len={text?.Length ?? 0})");
                }
            }

            if (filteredChunks.Count == 0)
            {
                Console.WriteLine("❌ No valid chunks to embed.");
                return;
            }

            Console.WriteLine($"\n🧠 Sending {filteredChunks.Count} texts to OpenAI in 2 batches for embedding...");

            int chunkSize = filteredTexts.Count / 3;
            var firstBatch = filteredTexts.Take(chunkSize).ToList();
            var secondBatch = filteredTexts.Skip(chunkSize).Take(chunkSize).ToList();
            var thirdBatch = filteredTexts.Skip(chunkSize * 2).ToList();

            Console.WriteLine($"🧠 Sending batch 1 with {firstBatch.Count} items...");
            OpenAIEmbeddingCollection firstResult = await client.GenerateEmbeddingsAsync(firstBatch);

            Console.WriteLine($"🧠 Sending batch 2 with {secondBatch.Count} items...");
            OpenAIEmbeddingCollection secondResult = await client.GenerateEmbeddingsAsync(secondBatch);

            Console.WriteLine($"🧠 Sending batch 3 with {thirdBatch.Count} items...");
            OpenAIEmbeddingCollection thirdResult = await client.GenerateEmbeddingsAsync(thirdBatch);

            var fullResult = firstResult.Concat(secondResult).Concat(thirdResult).ToList();

            for (int i = 0; i < fullResult.Count; i++)
            {
                filteredChunks[i].Embedding = fullResult[i].ToFloats().ToArray();
            }


            Console.WriteLine("✅ Embedding successful.");
        }
        catch (Exception ex)
        {
            Console.WriteLine("❌ Failed to generate embeddings.");
            Console.WriteLine($"🔍 Error: {ex.Message}");
        }


        string outputPath = GetProjectPath("Data/bgb_embeddings_new_format.jsonl");

        using (var writer = new StreamWriter(outputPath))
        {
            foreach (var chunk in allChunks)
            {
                string line = JsonSerializer.Serialize(chunk);
                writer.WriteLine(line);
            }
        }

        Console.WriteLine($"\n💾 Saved {allChunks.Count} chunks as JSONL to: {outputPath}");
    }

    private static string GetProjectPath(string relativePath)
    {
        string basePath = AppContext.BaseDirectory;
        string fullPath = Path.GetFullPath(Path.Combine(basePath, @"..\..\..", relativePath));
        return fullPath;
    }

    private static float CosineSimilarity(float[] a, float[] b)
    {
        float dot = 0f, normA = 0f, normB = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        return dot / ((float)Math.Sqrt(normA) * (float)Math.Sqrt(normB));
    }

    private static bool IsValidText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return false;

        text = text.Trim();

        if (text.Length < 10 || text.Length > 10000)
            return false;

        if (text.Any(c => char.IsControl(c) && c != '\n' && c != '\r'))
            return false;

        // Sometimes malformed HTML or unicode issues can result in invalid UTF-8
        try
        {
            Encoding.UTF8.GetByteCount(text);
        }
        catch
        {
            return false;
        }

        return true;
    }


}
