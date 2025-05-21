# 🧑‍⚖️ BGB Legal ChatBot (.NET Console App – Embedding Generator)

This is a C# console app designed to **parse the German Civil Code (BGB)** from HTML, **generate OpenAI embeddings**, and save them for use in legal question-answering via Retrieval-Augmented Generation (RAG).

This repo focuses on the **embedding generation** part of the pipeline.

> I initially prototyped querying here in C#, but later rewrote the full querying experience in Python + Gradio for flexibility and UI.  
> You can try the working bot here: [**Live BGB Chat Bot**](https://huggingface.co/spaces/nfm1708/BGBChatBot)  
> [**Source code for Python + Gradio Query Bot**](https://github.com/nida28/BGBPythonBot)

---

## 🚀 Features

- Parses and extracts legal paragraphs from a BGB HTML file
- Identifies and associates each paragraph with its section number and title
- Generates OpenAI embeddings using `text-embedding-3-small`
- Saves rich chunks with metadata + vector in `.jsonl`
- Prototype logic for querying is included (but moved to Python)

---
## 🧱 Tech Stack

- **.NET 8 Console App**
- **OpenAI SDK** (`OpenAI.Chat`, `OpenAI.Embeddings`)
- **HtmlAgilityPack** for HTML parsing
- **System.Text.Json** for fast JSON parsing
- **Manual cosine similarity search** (no vector DB needed)

## 📁 Project Structure
```bash
RAGBaseApp/
│
├── Program.cs // Main logic entry point
├── Models/
│ └── Chunk.cs // Defines Chunk structure with text, section, and embedding
├── Data/
│ ├── sample.html // Input BGB HTML source
│ └── bgb_embeddings_new_data.jsonl // Output embeddings as JSONL
```

## ⚙️ How to Run

1. **Set up your OpenAI key**

Set your API key in your environment variables (I used launchSettings.json in VS):

```bash
export OPENAI_API_KEY=your_key_here
```
2. **Choose mode**
   
By default, it runs PopulateEmbeddingsDatabase() to create embeddings.

To test the old C# query logic, you can uncomment StartLegalQueryLoop() in Main().

3. **Build and run the app**
```bash
dotnet build
dotnet run
```
## 📝 Notes
- Querying now lives in Python [**here**](https://github.com/nida28/BGBPythonBot)
- You can customize sample.html to include other html content - you will need to tweak the html parsing logic a bit.
- Embeddings are stored locally in JSONL for easy re-use
- Fully local RAG—no database required

## 📚 Future Ideas

- Vector DB integration (like Qdrant or Pinecone)
- Host as an API or Gradio demo
- Expand to other legal codes or languages
- Add metadata filters (e.g., search only tenancy laws)

## 🙌 Credits

Built with ❤️ for legal clarity, using:
- [OpenAI API](https://platform.openai.com/)
- [HtmlAgilityPack](https://html-agility-pack.net/)
- [German BGB](https://www.gesetze-im-internet.de/bgb/)

## 🧪 License

MIT — use it, remix it, expand it.

