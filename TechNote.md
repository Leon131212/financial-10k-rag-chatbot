# Tech Note — Financial 10-K RAG Chatbot

## 1. Approach

This project builds a document-grounded chatbot for answering questions about 10-K filings. The main goal was to improve reliability on financial questions by making the model answer from retrieved filing evidence rather than from unsupported prior knowledge.

My final approach uses a **RAG pipeline with retrieval, answer generation, and verification**. The workflow is:

1. Upload and parse 10-K PDFs  
2. Split documents into chunks with metadata  
3. Build a FAISS vector store  
4. Retrieve relevant chunks for the user query  
5. Generate an answer using only the retrieved context  
6. Verify whether the answer is supported by the evidence  
7. Regenerate if major issues are detected

This design was chosen because a plain chatbot was too likely to hallucinate, especially on financial statement questions and multi-company comparisons.

---

## 2. Model Choice, System Prompt, and Architecture

### Model choice
The system supports multiple LLMs and embedding models, including OpenAI, Gemini, and local Ollama models.  
For the final version, I used:

- **Primary generation model:** gpt-4o
- **Verifier model:** gemini-2.5-flash
- **Embedding model:** text-embedding-3-large

I chose a stronger model for generation because financial QA requires careful company attribution, line-item interpretation, and comparison reasoning.

### System prompt
The system prompt is designed to be **document-first**. It asks the model to:

- answer only from retrieved context
- distinguish between directly stated values and derived values
- avoid overriding a directly stated number
- avoid confusing subtotals with totals
- clearly say when a value is not available in the filing

This was important because a generic QA prompt often produced fluent but weakly grounded answers.

### Architecture
The architecture is a hybrid RAG system with:
- PDF ingestion
- metadata-aware chunking
- FAISS vector retrieval
- optional keyword retrieval and reranking
- answer generation
- verification and self-correction

This makes the system more robust than a simple upload-and-ask chatbot.

---

## 3. RAG Design

### Chunk size
The final chunk settings were:

- **Chunk size:** 1200  
- **Chunk overlap:** 200  

This was a balance between precision and context. Smaller chunks improved precision but sometimes lost table continuity, while larger chunks preserved context but added noise.

### Embedding model
The system converts each chunk into embeddings for semantic retrieval.  
Final embedding model used: text-embedding-3-large

### Vector store
I used FAISS as the vector store because it is efficient, easy to use locally, and integrates well with LangChain.

### Retrieval strategy
The system uses semantic retrieval as the main method, with optional enhancements such as:
- BM25 keyword retrieval
- MMR for diversity
- HyDE for difficult questions
- reranking

This was helpful because some financial questions depend on exact wording, while others need broader semantic matching.

---

## 4. Insights Gained and Failed Approaches

One major insight from this project is that retrieval alone is not enough. Earlier versions could retrieve relevant chunks but still fail in the final answer by:

- ignoring direct values in the filing
- using the wrong subtotal instead of the final total
- missing one company in comparison questions
- saying “not available” when the value could actually be found or derived

I also found several specific issues during development:

- Earlier versions sometimes discarded important statement chunks after retrieval because of scoring rules.
- Initial statement-pattern matching worked better for Alphabet and Amazon than for Microsoft, because Microsoft sometimes used different statement titles.
- Some pages were mistakenly classified as notes rather than statements.
- Earlier truncation settings sometimes removed critical financial lines.
- A generic generation prompt produced polished answers, but they were not always well grounded.

To fix these issues, later versions added:
- statement chunk pinning
- better statement-title matching
- improved metadata handling
- stronger prompt constraints
- verification and self-correction

The biggest lesson was that reliability in financial QA depends more on **retrieval coverage, chunk quality, and prompt discipline** than on fluent wording.

---

## 5. Strengths and Weaknesses

### Strengths
- More grounded than a plain chatbot
- Better at multi-company financial comparison
- Better handling of statement-based questions
- Reduced hallucination risk through verification
- Flexible support for different models and retrieval options

### Weaknesses
- PDF parsing is still imperfect, especially for complex tables
- Performance depends heavily on retrieval quality
- Verification is helpful but not perfect
- The pipeline is more complex and harder to tune than a basic chatbot
- Some financial questions are fundamentally unanswerable from a 10-K alone

---

## 6. Team Members and Roles

- **Chentong Xie** and **Yukun Song** — Primary developers; responsible for system architecture design, RAG implementation, retrieval improvement, verification design, debugging, and testing.

- **Xinkuo Qu** — Contributed to document preparation, prompt drafting, and review of test cases and outputs.

- **Qiman Zhang** — Assisted with usability feedback, result checking, discussion of model behavior, and presentation preparation.

- **Jianing Song** — Supported project coordination and helped review presentation and documentation materials.

---

## 7. Conclusion

This project showed that building a strong financial chatbot is not just about using a powerful model. The more important challenge is making the model stay faithful to the source document.

My final system improves reliability by combining chunked retrieval, vector search, metadata-aware processing, structured prompting, and verification. The most important insight I gained is that in financial document QA, a model can sound convincing and still be wrong, so system design matters as much as model choice.
