# DevBuddyBackend

Backend for **DevBuddy**. Contains a fully working CLI version as well as the FastAPI backend.

---

# DevBuddy 🧑‍💻📘

DevBuddy is your personal developer assistant that turns documentation websites into intelligent chatbots. Just provide a URL, and DevBuddy will:

- Crawl the site
- Convert its content into markdown
- Perform Retrieval-Augmented Generation (RAG)
- Answer your queries in real-time

---

## 🚀 Features

- 🔗 **URL-based Crawling** – Just give a documentation website URL, and DevBuddy handles the crawling automatically using [`crawl4ai`](https://github.com/AIxForce/crawl4ai).
- 🧠 **AI-Powered Chatbot** – Get contextual answers based on the documentation you provided.
- 📄 **Auto Markdown Conversion** – Extracted content is converted into clean markdown format for efficient indexing.
- 🔍 **RAG-based Search** – Combines document retrieval with generative AI for precise answers.
- 🧰 **Boilerplate Generator** – Ask DevBuddy for boilerplate code for any library/tool and get context-aware snippets.
- ⚡ **One-Click Setup** – No need to manually upload docs or markdowns.

---

## 🏗️ Tech Stack

- **Backend:** FastAPI
- **Crawling:** [`crawl4ai`](https://github.com/AIxForce/crawl4ai)
- **LLM:** Gemini 2.0 Flash (can be swapped with OpenAI or Claude)
- **Vector Store:** FAISS
- **Frontend:** React + TailwindCSS
- **RAG Engine:** Custom-built with chunking and semantic search

---

## 🧪 How It Works

1. **User provides a documentation website URL.**
2. DevBuddy uses `crawl4ai` to crawl the site using a BFS strategy with content filtering.
3. Markdown content is auto-generated using `DefaultMarkdownGenerator`.
4. A vector store is created from the markdown content.
5. The chatbot answers user queries via a RAG pipeline.

---

## ✨ Example Use Case

> "Hey DevBuddy, how do I set up authentication in Supabase?"

DevBuddy will return the precise snippet and explanation from Supabase's official documentation—no need to copy-paste or search manually.

---

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_key
```

## 🧠 Coming Soon

* 🔁 LLM switching between Gemini / OpenAI / Claude
* 🔍 Source highlighting in chatbot answers
* 🌐 Multi-site documentation merging and unified query answering

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open issues, submit pull requests, or suggest new features.

---

Built with ❤️ by [Sanjay Biju](https://github.com/sanislearning)
