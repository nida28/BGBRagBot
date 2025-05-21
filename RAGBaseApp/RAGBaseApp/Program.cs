using System.Text;
using System.Text.Json;
using HtmlAgilityPack;
using OpenAI.Chat;
using OpenAI.Embeddings;
using RAGBaseApp.Models;

class Program
{
    static async Task Main(string[] args)
    {
        await PopulateEmbeddingsDatabase();
        // await StartLegalQueryLoop();
    }

    private static async Task StartLegalQueryLoop()
    {
        string embeddingPath = GetProjectPath("Data/bgb_embeddings_new.json");

        if (!File.Exists(embeddingPath))
        {
            Console.WriteLine("No embeddings found.");
            return;
        }

        List<Chunk> allChunks = await LoadChunksFromDisk(embeddingPath);

        Console.Write("Enter your question: ");
        string question = Console.ReadLine();

        string apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            Console.WriteLine("OPENAI_API_KEY environment variable not found.");
            return;
        }

        ChatClient chatClient = new ChatClient("gpt-4o", apiKey);
        bool isLegalQuery = question.ToLower().Contains("section") || question.ToLower().Contains("bgb");

        if (isLegalQuery)
        {
            Console.WriteLine("BGB detected. Using local RAG...");

            EmbeddingClient embeddingClient = new EmbeddingClient("text-embedding-3-small", apiKey);
            OpenAIEmbedding queryEmbedding = await embeddingClient.GenerateEmbeddingAsync(question);
            float[] queryVector = queryEmbedding.ToFloats().ToArray();

            List<Chunk> topChunks = allChunks
                .Select(chunk => new { Chunk = chunk, Score = CosineSimilarity(queryVector, chunk.Embedding) })
                .OrderByDescending(x => x.Score)
                .Take(3)
                .Select(x => x.Chunk)
                .ToList();

            string prompt = BuildRAGPrompt(question, topChunks);
            ChatCompletion response = await chatClient.CompleteChatAsync(prompt);
            Console.WriteLine($"\nGPT (BGB grounded):\n{response.Content[0].Text}");
        }
        else
        {
            Console.WriteLine("💬 Sending directly to GPT...");
            ChatCompletion response = await chatClient.CompleteChatAsync(question);
            Console.WriteLine($"\nGPT says:\n{response.Content[0].Text}");
        }
    }

    private static string BuildRAGPrompt(string question, List<Chunk> topChunks)
    {
        StringBuilder promptBuilder = new StringBuilder();
        promptBuilder.AppendLine("You are a legal assistant. Use only the content below from the BGB.");
        promptBuilder.AppendLine();
        promptBuilder.AppendLine("When citing legal references, refer to the sections mentioned in the text, not the internal section IDs (e.g., p0001).");
        promptBuilder.AppendLine("Clearly explain tenant rights using plain, helpful language. Mention the section number (e.g., Section 535) where applicable.");
        promptBuilder.AppendLine();

        foreach (Chunk chunk in topChunks)
        {
            promptBuilder.AppendLine($"[Section: {chunk.Id}] {chunk.Text}");
            promptBuilder.AppendLine();
        }

        promptBuilder.AppendLine("---");
        promptBuilder.AppendLine($"Question: {question}");

        return promptBuilder.ToString();
    }

    private static async Task<List<Chunk>> LoadChunksFromDisk(string path)
    {
        List<Chunk> chunks = new List<Chunk>();
        using FileStream stream = File.OpenRead(path);

        await foreach (Chunk chunk in JsonSerializer.DeserializeAsyncEnumerable<Chunk>(stream, new JsonSerializerOptions
        {
            AllowTrailingCommas = true,
            PropertyNameCaseInsensitive = true
        }))
        {
            if (chunk != null)
            {
                chunks.Add(chunk);
            }
        }

        return chunks;
    }

    private static float CosineSimilarity(float[] a, float[] b)
    {
        float dotProduct = 0f;
        float normA = 0f;
        float normB = 0f;

        for (int i = 0; i < a.Length; i++)
        {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        return dotProduct / ((float)Math.Sqrt(normA) * (float)Math.Sqrt(normB));
    }

    private static string GetProjectPath(string relativePath)
    {
        string basePath = AppContext.BaseDirectory;
        return Path.GetFullPath(Path.Combine(basePath, @"..\..\..", relativePath));
    }

    private static bool IsValidText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        text = text.Trim();

        if (text.Length < 10 || text.Length > 10000)
        {
            return false;
        }

        if (text.Any(c => char.IsControl(c) && c != '\n' && c != '\r'))
        {
            return false;
        }

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

    private static async Task PopulateEmbeddingsDatabase()
    {
        string apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            Console.WriteLine("OPENAI_API_KEY not set.");
            return;
        }

        HtmlDocument document = LoadHtmlDocument("Data/sample.html");

        List<HtmlNode> contentNodes = document.DocumentNode
            .SelectNodes("//p[a[@name] and string-length(normalize-space()) > 10]")?
            .ToList();

        List<(string Id, int Position, string SectionNumber, string SectionTitle)> sectionHeaders = ParseSectionHeaders(document);

        if (contentNodes == null || contentNodes.Count == 0)
        {
            Console.WriteLine("No matching paragraph nodes found.");
            return;
        }

        Console.WriteLine($"Found {contentNodes.Count} content chunks.");
        Console.WriteLine($"Found {sectionHeaders.Count} section headers.");

        List<Chunk> allChunks = ExtractContentChunks(contentNodes, sectionHeaders);

        List<Chunk> filteredChunks = new List<Chunk>();
        List<string> textsToEmbed = new List<string>();

        for (int i = 0; i < allChunks.Count; i++)
        {
            string text = allChunks[i].Text;

            if (!string.IsNullOrWhiteSpace(text) && text.Length >= 10 && text.Length <= 10000)
            {
                filteredChunks.Add(allChunks[i]);
                textsToEmbed.Add(text.Trim());
            }
            else
            {
                Console.WriteLine($"Filtering out invalid text at index {i} (len={text?.Length ?? 0})");
            }
        }

        if (filteredChunks.Count == 0)
        {
            Console.WriteLine("No valid chunks to embed.");
            return;
        }

        Console.WriteLine($"\nSending {filteredChunks.Count} texts to OpenAI in batches...");

        List<OpenAIEmbedding> allEmbeddings = await GenerateEmbeddingsInBatches(apiKey, textsToEmbed);

        for (int i = 0; i < allEmbeddings.Count; i++)
        {
            filteredChunks[i].Embedding = allEmbeddings[i].ToFloats().ToArray();
        }

        SaveChunksAsJsonl(filteredChunks, "Data/bgb_embeddings_test.jsonl");
    }

    private static HtmlDocument LoadHtmlDocument(string relativePath)
    {
        HtmlDocument document = new HtmlDocument();
        string htmlPath = GetProjectPath(relativePath);
        document.Load(htmlPath);
        return document;
    }

    private static List<(string Id, int Position, string SectionNumber, string SectionTitle)> ParseSectionHeaders(HtmlDocument document)
    {
        List<(string Id, int Position, string SectionNumber, string SectionTitle)> sectionList = new List<(string, int, string, string)>();

        List<HtmlNode> headers = document.DocumentNode
            .SelectNodes("//p[a[contains(@class, 'section-anchor')]]")?
            .ToList();

        if (headers == null)
        {
            return sectionList;
        }

        foreach (HtmlNode node in headers)
        {
            HtmlNode anchor = node.SelectSingleNode(".//a");
            string id = anchor?.GetAttributeValue("name", null);
            int position = node.StreamPosition;

            string innerHtmlWithNewlines = node.InnerHtml
                .Replace("<br>", "\n")
                .Replace("<br/>", "\n")
                .Replace("<br />", "\n");

            HtmlDocument temp = new HtmlDocument();
            temp.LoadHtml(innerHtmlWithNewlines);

            string plainText = temp.DocumentNode.InnerText
                .Replace("\r", "\n")
                .Replace("\r\n", "\n");

            string[] lines = plainText
                .Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(s => s.Trim())
                .ToArray();

            string sectionNumber = lines.Length > 1 ? lines[1] : "";
            string sectionTitle = lines.Length > 2 ? string.Join(" ", lines.Skip(2)) : "";

            if (!string.IsNullOrEmpty(id) && !string.IsNullOrEmpty(sectionNumber))
            {
                sectionList.Add((id, position, sectionNumber, sectionTitle));
            }
        }

        return sectionList.OrderBy(s => s.Position).ToList();
    }

    private static List<Chunk> ExtractContentChunks(List<HtmlNode> nodes, List<(string Id, int Position, string SectionNumber, string SectionTitle)> sections)
    {
        List<Chunk> chunks = new List<Chunk>();

        foreach (HtmlNode node in nodes)
        {
            HtmlNode anchor = node.SelectSingleNode(".//a");
            string id = anchor?.GetAttributeValue("name", null);
            if (string.IsNullOrWhiteSpace(id)) continue;

            string rawText = HtmlEntity.DeEntitize(node.InnerText)
                .Replace("\r", " ")
                .Replace("\n", " ")
                .Trim();

            while (rawText.Contains("  ")) rawText = rawText.Replace("  ", " ");
            if (!IsValidText(rawText))
            {
                Console.WriteLine($"⚠️ Skipping chunk {id} (invalid text, len = {rawText.Length})");
                continue;
            }

            string sectionId = null;
            string sectionNumber = null;
            string sectionTitle = null;

            if (sections.Count > 0)
            {
                var nearestSection = sections.LastOrDefault(s => s.Position <= node.StreamPosition);
                if (!string.IsNullOrEmpty(nearestSection.Id))
                {
                    sectionId = nearestSection.Id;
                    sectionNumber = nearestSection.SectionNumber;
                    sectionTitle = nearestSection.SectionTitle;
                }
            }

            chunks.Add(new Chunk
            {
                Id = id,
                Text = rawText,
                SectionId = sectionId,
                SectionNumber = sectionNumber,
                SectionTitle = sectionTitle,
                Embedding = null
            });
        }

        return chunks;
    }

    private static async Task<List<OpenAIEmbedding>> GenerateEmbeddingsInBatches(string apiKey, List<string> texts)
    {
        List<OpenAIEmbedding> allEmbeddings = new List<OpenAIEmbedding>();
        EmbeddingClient client = new EmbeddingClient("text-embedding-3-small", apiKey);
        const int batchSize = 2000;

        for (int i = 0; i < texts.Count; i += batchSize)
        {
            List<string> batch = texts.Skip(i).Take(batchSize).ToList();
            Console.WriteLine($"Sending batch with {batch.Count} items (offset {i})...");

            OpenAIEmbeddingCollection result = await client.GenerateEmbeddingsAsync(batch);
            allEmbeddings.AddRange(result);
        }

        Console.WriteLine($"Embedding completed for {texts.Count} texts.");
        return allEmbeddings;
    }

    private static void SaveChunksAsJsonl(List<Chunk> chunks, string relativePath)
    {
        string outputPath = GetProjectPath(relativePath);

        using StreamWriter writer = new StreamWriter(outputPath);
        foreach (Chunk chunk in chunks)
        {
            string line = JsonSerializer.Serialize(chunk);
            writer.WriteLine(line);
        }

        Console.WriteLine($"\nSaved {chunks.Count} chunks as JSONL to: {outputPath}");
    }

}
