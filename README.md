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
```

## Key Design Decisions

### 1. Retrieval before generation
A core design goal was to reduce hallucinations in financial QA. Rather than relying on the model’s internal knowledge, the system retrieves document evidence first and only then generates an answer.

### 2. Chunked document representation
10-K filings are too long to send directly into a model context window in a reliable way. Chunking makes retrieval more precise and allows the model to reason over narrower, more relevant evidence.

### 3. Vector search with FAISS
FAISS provides a lightweight and effective local vector database for semantic search. It was a practical choice for rapid prototyping and for a project that can be run locally.

### 4. Prompting for grounded answers
The system prompt explicitly encourages the model to:
- answer only from retrieved context
- distinguish between directly disclosed facts and inferred estimates
- say when information is unavailable
- avoid treating proxies as facts

This is especially important in financial tasks where many seemingly simple questions cannot actually be answered directly from the filing.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Leon131212/financial-10k-rag-chatbot.git
cd financial-10k-rag-chatbot
```

On Windows:
```
  venv\Scripts\activate
```

  On macOS / Linux:
```
  source venv/bin/activate
```

  Install dependencies
```
  pip install -r requirements.txt
```
  ---
  Environment Variables

  Create a .env file in the project root and add the required API key(s).


## Running the Application

Launch the Streamlit app:
```
streamlit run app.py
```

## Evaluation Focus

This project was evaluated less as a generic chatbot and more as a document-grounded financial assistant. The key questions during development were:

- Does the system retrieve the right evidence?
- Does the answer stay faithful to the retrieved context?
- Can the model distinguish between direct disclosure and unsupported inference?
- Does the system avoid fabricated precision when filings do not disclose a metric directly?

These questions became especially important in tasks like:
- AI revenue comparisons across firms
- risk-factor comparisons
- extracting metrics that are not separately reported
- identifying whether a number is directly disclosed or only indirectly inferable

## Example Failure Mode the Project Addresses

A strong language model may produce an answer that looks polished and analytical but is not actually supported by the filing. For example, a model may:
- relabel segment revenue as “AI revenue”
- treat total capex as “AI capex”
- present proxy-based calculations as if they were directly disclosed
- overgeneralize risk factors across companies

This system is designed to reduce those errors by forcing the answering process to begin with retrieval and evidence, not fluent guessing.

## Limitations

This project improves reliability, but it does not eliminate all failure modes.
Current limitations include:
- PDF parsing can miss or distort tables
- retrieval quality depends heavily on chunk size and overlap
- weak retrieval can still lead to weak answers
- some financial questions are fundamentally unanswerable from the filings alone
- disclosure formats differ across companies, which makes direct comparison difficult
- segment-level proxies can be tempting but methodologically dangerous if not clearly labeled as assumptions

## What I Learned

This project reinforced an important lesson about applied LLM systems:

Better models do not automatically produce better document-based answers.

In financial QA, the key challenge is often not language generation but constraint. A model may sound intelligent while still drifting away from the source material. As a result, system design matters just as much as model choice.

Through this project, I gained hands-on experience with:

- building an end-to-end RAG pipeline
- designing a document-grounded QA workflow
- evaluating retrieval quality
- identifying hallucination risks in finance-oriented prompts
- distinguishing between direct evidence and model-generated inference

### Author
Chentong Xie
