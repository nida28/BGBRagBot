
using System;
using System.Linq;
using System.Threading.Tasks;
using HtmlAgilityPack;
using OpenAI;
using OpenAI.Embeddings;

class Program
{
    static async Task Main(string[] args)
    {
        // ✅ Load API key from environment variable
        string apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");

        if (string.IsNullOrWhiteSpace(apiKey))
        {
            Console.WriteLine("❌ OPENAI_API_KEY environment variable not found.");
            return;
        }

        EmbeddingClient client = new("text-embedding-3-small", Environment.GetEnvironmentVariable("OPENAI_API_KEY"));

        // ✅ Load the HTML file
        HtmlDocument doc = new HtmlDocument();
        doc.Load("Data/bgb.html");

        // ✅ Select legal paragraph chunks
        var nodes = doc.DocumentNode.SelectNodes("//div[contains(@class, 'jurAbsatz')]");

        if (nodes == null || nodes.Count == 0)
        {
            Console.WriteLine("❌ No <div class='jurAbsatz'> elements found.");
            return;
        }

        Console.WriteLine($"🔍 Found {nodes.Count} legal paragraphs...");

        // ✅ Embed a few chunks (start with 3 to avoid high usage)
        foreach (var node in nodes.Take(2))
        {
            string text = node.InnerText.Trim();

            if (string.IsNullOrWhiteSpace(text)) continue;

            text = HtmlEntity.DeEntitize(text);
            text = text.Replace("\r", " ").Replace("\n", " ");
            while (text.Contains("  ")) text = text.Replace("  ", " ");
            text = text.Trim();

            if (string.IsNullOrWhiteSpace(text) || text.Length < 10)
            {
                Console.WriteLine("⚠️ Skipping blank or short chunk.");
                continue;
            }

            if (text.Length > 10000)
            {
                Console.WriteLine("⚠️ Skipping or trimming overly long chunk.");
                text = text.Substring(0, 10000); // or use a smarter tokenizer later
            }


            OpenAIEmbedding embedding = client.GenerateEmbedding(text);
            ReadOnlyMemory<float> vector = embedding.ToFloats();

            Console.WriteLine($"Dimension: {vector.Length}");
            Console.WriteLine($"Floats: ");
            for (int i = 0; i < vector.Length; i++)
            {
                Console.WriteLine($"  [{i,4}] = {vector.Span[i]}");
            }
        }

        Console.WriteLine("\n🎉 Done!");
    }
}
