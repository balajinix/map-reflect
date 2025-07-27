# 📄 API Documentation Processing and Lookup Pipeline

This repository contains tools for extracting, enhancing, and storing structured API metadata from web documentation of **Concert**, **Instana**, and **Turbonomic**.

The goal is to:
- Web scrape official API documentation pages.
- Chunk content and store it in a vector database (ChromaDB).
- Use RAG (retrieval augmented generation) to create better endpoint descriptions.
- Populate a unified API lookup JSON file.

---

## 📁 Folder Structure
```
docs_info/
├── concert_api_docs/
│   ├── get_page_links.py
│   ├── web_scrape.py
│   ├── create_descriptions.py
│   ├── api_processor.py
│   └── get_info.ipynb
├── instana_turbo_api_docs/
│   ├── IBMTurbonomic_API_ep.pdf
│   ├── turbo_get_info.ipynb
│   ├── instana_get_info.ipynb
│   ├── create_descriptions.py
│   └── api_processor.py
├── create_all_mappings.py
├── final_populate_lookup.py
├── .env
└── README.md  <-- You are here!
```

---

## 🧠 Overview

(Please install Chrome Driver to run the webscraping scripts that use selenium)

### 🔹 `concert_api_docs/`

- **`get_page_links.py`**  
  Scrapes the Concert documentation hub and gathers links to relevant API pages.

- **`web_scrape.py`**  
  Visits each gathered link, extracts HTML/visible content, chunks it, and stores it in a **ChromaDB collection** called `concert_api`.

- **`create_descriptions.py`**  
  Uses RAG over `concert_api` to generate improved field descriptions for each API endpoint.

- **`api_processor.py`**  
  Processes schema JSON files, calls the description generator, and prepares structured metadata for the lookup store.

- **`get_info.ipynb`**  
  Jupyter notebook for experimentation — contains combined scraping, chunking, and ChromaDB insertion workflows.

---

### 🔹 `instana_turbo_api_docs/`

- **`IBMTurbonomic_API_ep.pdf`**  
  Reference PDF for the Turbonomic API.

- **`instana_get_info.ipynb`**  
  Processes and chunks Instana's API documentation into ChromaDB.

- **`turbo_get_info.ipynb`**  
  Processes and chunks Turbonomic's API documentation into ChromaDB.

- **`create_descriptions.py`**  
  Similar to Concert, but supports **routing** between Instana and Turbo ChromaDB collections.

- **`api_processor.py`**  
  Handles schema parsing and metadata enrichment using the correct backend store (Instana or Turbo).

---

### 🔹 `final_populate_lookup.py`

> **Main script to build the final API lookup store**

- Traverses all input schema JSON files.
- Determines API source (Concert, Instana, Turbo).
- Fetches enriched descriptions using RAG.
- Writes results to a unified `api_lookup.json`.

---

### 🔹 `create_all_mappings.py`

> **Script to generate schema mappings between every pair of input schemas**

- Loads all schemas from a given directory.
- Uses the Schema Mapping Toolkit to compute all possible mappings.

---