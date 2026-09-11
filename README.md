# Enterprise Smart Knowledge Engine (Semantic RAG System)

A high-performance **Retrieval-Augmented Generation (RAG)** pipeline built using **Node.js**, **LangChain.js**, and **Google Gemini-3.6-Flash** [rag.js]. This project demonstrates how to ingest unstructured text documents, vectorize data segments, and orchestrate secure semantic information retrieval.

## 🧠 Core Architecture & Pipelines
* **Document Ingestion:** Utilized native file system streams to parse unstructured corporate documents [rag.js].
* **Text Chunk Splitting:** Implemented `RecursiveCharacterTextSplitter` to break long paragraphs into optimized text vectors with calculated boundaries to retain semantic context [rag.js].
* **Vector Embeddings:** Generated mathematical definitions using Google's `text-embedding-004` architecture [rag.js].
* **Vector Database Cache:** Stored context blocks inside an in-memory vector store for high-speed similarity search querying [rag.js].
* **Context Injected Chains:** Designed a strict **LCEL (LangChain Expression Language)** pipeline that extracts matching documents and injects them as raw context buffers into the final prompt to eradicate model hallucinations [rag.js].

## ⚙️ How It Works
1. A raw text document (`knowledge.txt`) is read and split into overlapping text chunks [rag.js].
2. The chunks are converted into multi-dimensional vectors and cached in a searchable store [rag.js].
3. When a query lands, the system runs a **mathematical similarity search** to isolate the target text paragraph matching the context [rag.js].
4. The exact background paragraph is injected directly into the LLM system prompt, forcing the model to answer accurately based *only* on the verified source material [rag.js].

## 🚀 Local Execution
1. Install project package dependencies:
   ```bash
   npm install
   ```
2. Configure your environment credential key inside a `.env` file:
   ```env
   GOOGLE_API_KEY="your-google-ai-studio-api-key"
   ```
3. Run the evaluation script:
   ```bash
   node rag.js
   ```
