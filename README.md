# Rag_Agent

## Project Architecture

This project implements a Retrieval-Augmented Generation (RAG) agent for nutrition facts Q&A. The architecture consists of several main modules:

- **Web Scraper (`simple_rag_app/web_scraper.py`)**: Scrapes blog posts and nutrition articles, saving them as JSON files for ingestion.
- **Data Ingestion (`simple_rag_app/data_ingestion.py`)**: Processes and chunks scraped data, embeds text using Sentence Transformers, and stores it in LanceDB with vector and FTS indexes.
- **Retrieval (`simple_rag_app/retrieval.py`)**: Performs hybrid search (vector + FTS) and reranks results using a cross-encoder. Groups and enriches context for user queries.
- **RAG Flow (`simple_rag_app/rag_flow.py`)**: Orchestrates the retrieval and LLM query flow, building prompts and interacting with Gemini LLM.
- **Streamlit App (`streamlit_app.py`)**: Provides a chat-based UI for users to ask nutrition questions, displaying answers and retrieved context.

For a detailed diagram and explanation of the architecture, see [Architectures.pdf](./Architectures.pdf).

---

*For more details on each module, see the respective Python files and the `simple_rag_app/README.md`.*