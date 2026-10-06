# Multi-Paper Research RAG

## Overview

This project implements a multi-paper Retrieval-Augmented Generation (RAG) system for querying research papers. It uses the `all-MiniLM-L6-v2` model from the SentenceTransformers library to create embeddings of text chunks extracted from the provided PDF research papers.

For a user's query, the system uses cosine similarity to retrieve the top 3 most relevant chunks across all papers. These chunks, along with their paper names and page numbers, are used as context to construct a prompt for an LLM. The LLM then generates a concise answer grounded in the retrieved context, with citations to the relevant paper and page.

## Architecture

PDF Research Papers
↓
Text Extraction
↓
Page-aware Chunking
↓
MiniLM Embeddings
↓
Cosine Similarity Retrieval
↓
Top-3 Relevant Chunks
↓
Prompt Construction
↓
LLM Generation
↓
Answer with Paper + Page Citations

- **PDF extraction:** PyMuPDF
- **Chunking:** 1000 characters with 200-character overlap, performed page-by-page
- **Embedding model:** `all-MiniLM-L6-v2`
- **Embedding dimension:** 384
- **Retrieval:** cosine similarity
- **Retrieved context:** top 3 chunks
- **Generation model:** `nvidia/nemotron-3-super-120b-a12b:free` through OpenRouter
- **Grounding:** the LLM is instructed to answer only from retrieved context and cite the paper and page

## Retrieval Evaluation

The retrieval system was evaluated using a manually curated benchmark of 20 questions across four research papers, with 5 questions corresponding to each paper.

For every question, the expected source paper was specified and the top 5 chunks were retrieved using cosine similarity. Paper-level Recall@K measures whether at least one chunk from the expected paper appears within the top K retrieved chunks.

| Metric | Score |
|---|---:|
| Recall@1 | 0.80 |
| Recall@3 | 1.00 |
| Recall@5 | 1.00 |

The correct paper was therefore retrieved as the highest-ranked result for 80% of the evaluation questions and appeared within the top 3 results for all 20 questions.

### Failure Analysis

The four Recall@1 failures primarily occurred between papers covering semantically related manipulation and learning concepts. In all four cases, the expected paper was still retrieved within the top 3 results.

This suggests that the baseline MiniLM retriever successfully identifies relevant papers but can confuse the highest-ranked result when multiple papers discuss similar concepts. Possible future improvements include reranking, hybrid retrieval, and evaluating alternative embedding models.

## Example

Run the RAG system:

```bash
python src/main.py
```

Example query:

```text
How does Neural MP perform motion planning?
```

Example answer:

```text
Status Code: 200
Model name: nvidia/nemotron-3-super-120b-a12b:free

Answer:
Neural MP carries out motion planning by executing a learned neural policy in a closed-loop fashion, continually observing the environment and issuing motion commands step-by-step. When faced with dynamic obstacles, it refines its policy online with single-step test-time optimization, allowing it to adjust its trajectory to avoid collisions while still progressing toward the goal【Neural_MP, Page 7】.
```

## Installation and Setup

Clone the repository and move into the project directory:

```bash
git clone https://github.com/adityajain001/research-paper-rag.git
cd research-paper-rag
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your OpenRouter API key:

```text
OPENROUTER_API_KEY=your_api_key_here
```

Place the research paper PDFs inside the `data/` directory.

Run the RAG system:

```bash
python src/main.py
```

To run the retrieval evaluation:

```bash
python src/evaluate.py
```
## Project Structure

```text
research-paper-rag/
├── data/
│   ├── factr.pdf
│   ├── VIPRA.pdf
│   ├── ManipGen.pdf
│   └── Neural_MP.pdf
├── src/
│   ├── __init__.py
│   ├── ingest.py
│   ├── rag.py
│   ├── main.py
│   └── evaluate.py
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

- `ingest.py` — extracts text from PDFs, creates page-aware chunks, and generates embeddings.
- `rag.py` — performs retrieval, constructs the context and prompt, and calls the LLM.
- `main.py` — runs the complete RAG pipeline for a user's query.
- `evaluate.py` — evaluates retrieval performance using the 20-question benchmark.
- `data/` — contains the research papers used by the system.

## Limitations and Future Work

The current system uses dense embedding-based retrieval with a fixed chunking strategy. While it achieves 100% Recall@3 on the current evaluation set, the benchmark contains only 20 manually curated questions across four papers.

Possible future improvements include:
- Hybrid retrieval combining semantic and keyword search
- Reranking retrieved chunks before generation
- Evaluation with a larger and more diverse benchmark
- Comparison of different embedding models
- Support for larger research-paper collections