# Multi-Paper Research RAG

A multi-paper Retrieval-Augmented Generation (RAG) system for asking natural-language questions across a collection of research papers.

The system retrieves relevant passages using semantic similarity and generates grounded answers using an LLM, with citations to the source paper and page number.

The project includes both a command-line pipeline and a Streamlit web application where users can upload their own PDFs and query them directly.

## Live Demo

https://multi-paper-rag.streamlit.app
---

## Features

- Upload multiple research papers in PDF format
- Page-aware PDF text extraction using PyMuPDF
- Overlapping text chunking with paper and page metadata
- Semantic embeddings using `all-MiniLM-L6-v2`
- Cosine-similarity retrieval across multiple papers
- Top-k context construction
- LLM-based answer generation through OpenRouter
- Paper and page citations in generated answers
- Streamlit web interface for interactive querying
- Session-based reuse of processed chunks and embeddings
- Retrieval evaluation using Recall@K

The pipeline is domain-agnostic and can be used with research papers from different fields. The current evaluation corpus consists of four robot-learning papers.

---

## Architecture

```text
User uploads PDFs
        ↓
PyMuPDF text extraction
        ↓
Page-aware overlapping chunking
        ↓
SentenceTransformer embeddings
(all-MiniLM-L6-v2)
        ↓
Cosine-similarity retrieval
        ↓
Top-k relevant chunks
        ↓
Context + question prompt
        ↓
LLM through OpenRouter
        ↓
Grounded answer
        ↓
Paper + page citations
```

For the Streamlit application, processed chunks and embeddings are stored in session state so that asking additional questions over the same uploaded papers does not require reprocessing the entire corpus.

---

## Retrieval Evaluation

The retriever was evaluated using 20 manually curated questions across four robot-learning research papers.

Each question was associated with an expected source paper, and retrieval was considered successful when a chunk from the expected paper appeared within the top-K retrieved chunks.

| Metric | Score |
|---|---:|
| Recall@1 | 0.80 |
| Recall@3 | 1.00 |
| Recall@5 | 1.00 |

This means:

- 16 out of 20 questions retrieved the expected paper at rank 1.
- All 20 questions retrieved the expected paper within the top 3.
- All 20 questions retrieved the expected paper within the top 5.

The results suggest that the embedding-based retriever reliably identifies the relevant region of the corpus, even when semantically related papers compete for the highest-ranked result.

---

## Failure Analysis

The four Recall@1 failures were primarily caused by semantic overlap between papers discussing related robot-learning and manipulation concepts.

Importantly, the expected paper still appeared within the top three results for every evaluation question.

In other words:

> The retriever knew the right neighborhood, even when it didn't pick the right house.

This is why the generation pipeline uses multiple retrieved chunks rather than relying exclusively on the highest-ranked result.

---

## Example

Question:

```text
How does Neural MP perform motion planning around obstacles?
```

Example output:

```text
Status Code: 200
Model name: nvidia/nemotron-3-super-120b-a12b:free

Answer:
Neural MP carries out motion planning by executing a learned neural policy
in a closed-loop fashion, continually observing the environment and issuing
motion commands step-by-step. When faced with dynamic obstacles, it refines
its policy online with single-step test-time optimization, allowing it to
adjust its trajectory to avoid collisions while still progressing toward
the goal【Neural_MP, Page 7】.
```

---

## Streamlit Web App

The web interface allows users to upload their own research papers instead of manually placing PDFs inside the project directory.

The workflow is:

```text
Upload PDFs
    ↓
Process and embed papers
    ↓
Enter a question
    ↓
Retrieve relevant chunks
    ↓
Generate grounded answer
    ↓
Display answer with citations
```

When the uploaded paper set remains unchanged, the application reuses the existing chunks and embeddings across questions.

If papers are added or removed, the corpus and embeddings are rebuilt for the new set of documents.

---

## Project Structure

```text
research-paper-rag/
│
├── src/
│   ├── app.py          # Streamlit web application
│   ├── main.py         # Command-line RAG pipeline
│   ├── ingest.py       # PDF extraction, chunking, and embeddings
│   └── rag.py          # Retrieval, prompt construction, and generation
│
├── data/               # Local research PDFs (not committed)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/adityajain001/research-paper-rag.git
cd research-paper-rag
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## OpenRouter Setup

The generation model is accessed through the OpenRouter API.

For local use, create a `.env` file in the project root:

```text
OPENROUTER_API_KEY=your_api_key_here
```

The `.env` file is excluded from Git and should never be committed to the repository.

---

## Run the Web App

Start the Streamlit application:

```bash
streamlit run src/app.py
```

Then upload one or more PDF research papers through the browser interface and ask questions about them.

---

## Run the Command-Line Version

Place research PDFs inside the `data/` directory and run:

```bash
python src/main.py
```

The command-line application will process the papers and prompt you to enter a question.

---

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- SentenceTransformers
- `all-MiniLM-L6-v2`
- scikit-learn
- OpenRouter API
- LLM-based grounded generation

---

## Current Limitations

- Character-based chunking rather than structure-aware or token-aware chunking
- Brute-force cosine-similarity search rather than a vector database
- Retrieval evaluation measures expected-paper retrieval rather than exact passage relevance
- Generated answers depend on the quality of retrieved context
- PDF text extraction quality depends on the structure and encoding of the uploaded document
- No persistent document storage; uploaded papers are intended for the active application session
- No conversational memory between questions

---

## Possible Future Improvements

- Token-aware or semantic chunking
- Vector database integration for larger document collections
- Reranking retrieved passages
- More detailed passage-level retrieval evaluation
- Automated answer-quality evaluation
- Support for additional document formats
- Conversational follow-up questions