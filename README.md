# financial-10k-rag-chatbot
A RAG-based chatbot for question answering over 10-K filings.

# Financial 10-K RAG Chatbot

A document-grounded chatbot for answering financial questions from 10-K filings using Retrieval-Augmented Generation (RAG).

## Overview

This project is a financial question-answering system built to help users interact with long-form corporate filings more efficiently. Instead of manually searching through hundreds of pages of annual reports, users can ask natural-language questions and receive answers grounded in the underlying 10-K documents.

The chatbot is designed around a simple but important principle:

> **Do not answer from memory when the answer should come from the filing.**

To support that goal, the system combines PDF ingestion, semantic chunking, vector retrieval, and LLM-based answer generation. It is especially useful for evidence-based tasks such as:

- extracting financial metrics from annual reports
- comparing disclosures across companies
- identifying shared vs. company-specific risks
- summarizing management disclosures
- reducing unsupported answers in financial QA workflows

This project was built as both a practical RAG system and an experiment in improving factual reliability for financial-document analysis.

---

## Demo Use Cases

Example questions the system is intended to handle:

- What was Microsoft’s total revenue in fiscal year 2024?
- What risks are shared by Amazon, Alphabet, and Microsoft based on their latest 10-K filings?
- Which company appears most dependent on advertising revenue?
- What does Alphabet disclose about uncertain tax positions?
- Compare cloud-related disclosures across the three filings.
- Is a given metric directly disclosed, or would calculating it require an unsupported assumption?

---

## Why This Project Matters

Financial filings are information-rich but difficult to query efficiently. Standard LLMs often produce confident answers even when a filing does not explicitly support them. In a high-stakes domain like finance, that behavior is problematic.

This project addresses that challenge by introducing a retrieval layer between the user query and the model response. Instead of asking the model to “know” the answer, the system first retrieves relevant evidence and then asks the model to answer using that context.

The result is a workflow that is more transparent, more auditable, and better aligned with how document-based financial analysis should work.

---

## System Architecture

The chatbot follows a standard RAG pipeline:

1. **Document ingestion**  
   10-K filings are loaded from PDF files.

2. **Chunking**  
   Long documents are split into overlapping text chunks for retrieval.

3. **Embedding**  
   Each chunk is converted into a semantic vector representation.

4. **Vector indexing**  
   Embeddings are stored in a FAISS vector store for fast similarity search.

5. **Retrieval**  
   At query time, the system retrieves the most relevant chunks from the filings.

6. **Grounded generation**  
   The retrieved context is passed to an LLM with instructions to answer only from the provided evidence and avoid unsupported claims.

---

## Tech Stack

- **Python**
- **Streamlit** for the user interface
- **LangChain** for document processing and retrieval orchestration
- **PyPDFLoader** for PDF ingestion
- **RecursiveCharacterTextSplitter** for chunking
- **OpenAI Embeddings** or equivalent embedding model
- **FAISS** for vector similarity search
- **OpenAI / Gemini / other LLM** for answer generation

---

## Repository Structure

```text
financial-10k-rag-chatbot/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_10k_files/
├── vector_store/
├── utils/
│   ├── loader.py
│   ├── retriever.py
│   ├── prompts.py
│   └── preprocess.py
└── notebooks/
    └── experiments.ipynb
