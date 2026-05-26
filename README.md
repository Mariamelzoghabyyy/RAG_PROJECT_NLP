# Grand Egyptian Museum — Bilingual RAG AI Assistant

<div align="center">

![GEM](https://img.shields.io/badge/Grand%20Egyptian%20Museum-AI%20Assistant-gold?style=for-the-badge)
![Arabic](https://img.shields.io/badge/Arabic-Native%20Support-green?style=for-the-badge)
![English](https://img.shields.io/badge/English-Native%20Support-blue?style=for-the-badge)
![Modal](https://img.shields.io/badge/Deployed%20on-Modal-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

### A Production-Ready Bilingual Retrieval-Augmented Generation (RAG) Assistant for the Grand Egyptian Museum

Ask questions in Arabic or English and receive grounded, source-aware answers powered by semantic retrieval and large language models.

[Live Demo](#) • [Report Bug](#) • [Request Feature](#)

</div>

---

# Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Why RAG Instead of a Plain LLM?](#why-rag-instead-of-a-plain-llm)
- [Technology Stack](#technology-stack)
- [Models Used](#models-used)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [API Usage](#api-usage)
- [Evaluation](#evaluation)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [Technical Notes](#technical-notes)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

# Overview

This project implements a bilingual Retrieval-Augmented Generation (RAG) AI assistant for the official website of the Grand Egyptian Museum ([gem.eg](https://gem.eg)).

The assistant combines:
- multilingual semantic retrieval,
- vector similarity search,
- grounded prompting,
- and large language model generation

to provide accurate museum-specific responses in both Arabic and English.

Unlike a standard LLM chatbot, the system retrieves information directly from museum content before generating answers, significantly reducing hallucinations and improving factual reliability.

---

# Highlights

- Native Arabic and English support
- Automatic language detection
- Cross-lingual semantic retrieval
- Source-grounded answer generation
- Dynamic relevance filtering
- Query-aware prompting
- Semantic vector search using FAISS
- Retrieval-based hallucination reduction
- Serverless cloud deployment using Modal
- Source citations with similarity scores

---

# Example Queries

## English Example

```text
Visitor:
"What are the opening hours of the Grand Egyptian Museum?"

Assistant:
"The Grand Egyptian Museum is open daily from 9 AM to 6 PM,
with extended hours until 9 PM on Saturdays and Wednesdays."

Source:
Plan Your Visit (gem.eg) — Similarity: 0.852
```

## Arabic Example

```text
الزائر:
"أخبرني عن قناع توت عنخ آمون الذهبي"

المساعد:
"قناع الدفن الذهبي لتوت عنخ آمون هو أحد أشهر القطع الأثرية
في العالم. وُجد القناع فوق رأس الملك المحنط..."

المصدر:
قناع الدفن الذهبي — Similarity: 0.865
```

---

# System Architecture

The system follows a Retrieval-Augmented Generation (RAG) pipeline optimized for bilingual museum question answering.

```text
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                       │
│         Streamlit Web Application (Frontend)           │
└─────────────────────────┬───────────────────────────────┘
                          │
                          │ HTTP Request
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  MODAL API ENDPOINT                    │
│                 FastAPI + RAG Pipeline                 │
│                                                         │
│ 1. Language Detection                                   │
│    → Arabic / English classification                    │
│                                                         │
│ 2. Query Classification                                 │
│    → visitor_info | history | general                   │
│                                                         │
│ 3. Query Embedding                                      │
│    → multilingual-e5-base (768-dimensional vectors)     │
│                                                         │
│ 4. Vector Retrieval                                     │
│    → FAISS similarity search (top-k chunks)             │
│                                                         │
│ 5. Dynamic Filtering                                    │
│    → relevance thresholding                             │
│                                                         │
│ 6. Context Truncation                                   │
│    → token-window protection                            │
│                                                         │
│ 7. Prompt Construction                                  │
│    → language-aware grounded prompting                  │
│                                                         │
│ 8. LLM Generation                                       │
│    → grounded multilingual response                     │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    VECTOR DATABASE                      │
│                    Modal Volume: rag                    │
│                                                         │
│ • FAISS vector index                                    │
│ • Chunk metadata                                        │
│ • Precomputed embeddings                                │
│ • Cached embedding model                                │
└─────────────────────────────────────────────────────────┘
```

---

# Why RAG Instead of a Plain LLM?

| Feature | Plain LLM | This RAG System |
|---|---|---|
| Museum-specific knowledge | Hallucinated or generic | Retrieved from trusted museum sources |
| Opening hours and visitor information | Potentially outdated | Retrieved from live scraped content |
| Source grounding | None | Retrieval-based generation |
| Arabic language support | Generic multilingual support | Optimized bilingual retrieval |
| Knowledge freshness | Limited by training cutoff | Continuously updateable |
| Hallucination risk | Higher | Reduced through retrieval grounding |
| Source transparency | None | Retrieved sources with similarity scores |

---

# Technology Stack

| Layer | Technology | Role |
|---|---|---|
| Cloud Infrastructure | Modal | Serverless deployment, GPU execution, persistent storage |
| Frontend | Streamlit | Interactive bilingual chat interface |
| API Layer | FastAPI | Inference and retrieval endpoint |
| Web Scraping | Playwright + BeautifulSoup4 | JavaScript-rendered content extraction |
| Text Processing | regex + custom cleaning pipeline | Bilingual normalization and cleanup |
| Chunking | LangChain TextSplitter + tiktoken | Token-aware document segmentation |
| Embedding Model | `intfloat/multilingual-e5-base` | Multilingual semantic embeddings |
| Vector Database | FAISS (`IndexFlatIP`) | High-speed cosine similarity retrieval |
| LLM Inference | Groq + Llama 3.1 8B Instant | Grounded response generation |
| Evaluation Framework | Custom RAG Evaluator | Retrieval and answer quality evaluation |

---

# Models Used

## multilingual-e5-base (Embedding Model)

| Property | Value |
|---|---|
| Model | `intfloat/multilingual-e5-base` |
| Parameters | 278 Million |
| Embedding Size | 768 Dimensions |
| Languages | 100+ |
| License | MIT |
| Purpose | Multilingual semantic retrieval |

### Important Notes

The E5 model requires prefixes during embedding generation:

```text
Documents → "passage: [content]"
Queries   → "query: [question]"
```

The model supports cross-lingual retrieval, allowing Arabic questions to retrieve English content and vice versa.

---

## Llama 3.1 8B Instant (LLM)

| Property | Value |
|---|---|
| Provider | Groq |
| Parameters | 8 Billion |
| Context Window | 128K Tokens |
| Purpose | Grounded multilingual response generation |

The model responds in the same language as the user's query.

---

# Project Structure

```text
GEM-RAG/
│
├── DATA PIPELINE
│   ├── gem_scraper.py
│   ├── gem_cleaner.py
│   ├── gem_chunker.py
│   ├── gem_embedder.py
│   ├── gem_faiss_builder.py
│   └── gem_retriever.py
│
├── RAG PIPELINE
│   └── gem_pipeline.py
│
├── FRONTEND
│   └── gem_streamlit.py
│
├── UTILITIES
│   ├── gem_prompt_templates.py
│   ├── gem_evaluator.py
│   └── gem_run_all.py
│
├── requirements.txt
└── README.md
```

### Modal Volume Structure

```text
rag/
├── all_documents.json
├── cleaned_documents.json
├── all_chunks.json
├── embeddings.npy
├── faiss_index.bin
├── chunks_metadata.json
└── model_cache/
```

---

# Quick Start

## Prerequisites

```bash
python --version
pip install modal
modal setup
pip install streamlit requests tiktoken
```

---

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/gem-rag-assistant.git
cd gem-rag-assistant
```

---

## 2. Configure Secrets

Create a Groq API key from:

https://console.groq.com/keys

Then configure Modal secrets:

```bash
modal secret create gem-secrets \
  OPENAI_API_KEY=gsk_your_groq_key_here \
  OPENAI_BASE_URL=https://api.groq.com/openai/v1 \
  LLM_MODEL=llama-3.1-8b-instant
```

---

## 3. Run the Full Pipeline

```bash
python gem_run_all.py
```

Or run each stage individually:

```bash
modal run gem_scraper.py
modal run gem_cleaner.py
modal run gem_chunker.py
modal run gem_embedder.py
modal run gem_faiss_builder.py
modal run gem_retriever.py
```

---

## 4. Test the Pipeline

```bash
modal run gem_pipeline.py
```

Example output:

```text
Question:
"What are the opening hours of the Grand Egyptian Museum?"

Language:
EN

Type:
visitor_info

Answer:
"The Grand Egyptian Museum is open daily from 9 AM to 6 PM..."

Retrieved Sources:
[0.852] Plan Your Visit
```

---

## 5. Deploy the API

```bash
modal deploy gem_pipeline.py
```

Example endpoint:

```text
https://yourname--gem-ask.modal.run
```

---

## 6. Launch the Streamlit Interface

Update the endpoint URL inside `gem_streamlit.py`, then run:

```bash
streamlit run gem_streamlit.py
```

The application launches locally at:

```text
http://localhost:8501
```

---

# Pipeline Stages

## 1. Web Scraping

`gem_scraper.py`

Scrapes museum pages using Playwright and BeautifulSoup4.

### Features

- JavaScript-rendered page extraction
- Retry handling with exponential backoff
- URL checkpointing and resume support
- Arabic and English page detection
- Noise removal using CSS selectors

---

## 2. Text Cleaning

`gem_cleaner.py`

Normalizes and cleans multilingual museum content.

### Features

- Unicode normalization
- Arabic diacritic removal
- Alef normalization
- Duplicate line filtering
- Language-specific cleanup rules

---

## 3. Document Chunking

`gem_chunker.py`

Splits documents into retrievable token-aware chunks.

### Configuration

```python
chunk_size     = 400
chunk_overlap  = 80
min_chunk_size = 30
```

### Features

- Heading-aware segmentation
- Language-aware separators
- Metadata-rich chunk generation
- URL-based chunk identifiers

---

## 4. Embedding Generation

`gem_embedder.py`

Converts chunks into semantic vectors using multilingual-e5-base.

### Configuration

```text
Embedding Size: 768
Normalization: L2
Device: GPU
Batch Size: 128
```

---

## 5. FAISS Index Construction

`gem_faiss_builder.py`

Builds the vector similarity index used during retrieval.

### Configuration

```text
Index Type: IndexFlatIP
Search Metric: Cosine Similarity
```

---

## 6. Retrieval Evaluation

`gem_retriever.py`

Tests cross-lingual semantic retrieval performance.

### Sample Queries

```text
EN → "What are the opening hours?"
EN → "Tell me about Tutankhamun's mask"
AR → "ما هي مواعيد فتح المتحف؟"
AR → "قناع توت عنخ آمون الذهبي"
```

---

# Configuration

## Retrieval Parameters

```python
top_k           = 5
score_threshold = 0.30
dynamic_floor   = max(0.30, best_score * 0.60)
```

## LLM Parameters

```python
temperature = 0.7
max_tokens  = 1200
```

---

# Deployment

## Backend API Deployment

```bash
modal deploy gem_pipeline.py
```

Generated endpoints:

```text
POST https://yourname--gem-ask.modal.run
GET  https://yourname--gem-health.modal.run
```

---

## Streamlit Cloud Deployment

1. Push the repository to GitHub
2. Open Streamlit Cloud
3. Connect the repository
4. Configure environment variables

```text
MODAL_ASK_URL
MODAL_HEALTH_URL
```

5. Deploy the application

---

# API Usage

## Example Request

```python
import requests

response = requests.post(
    "https://yourname--gem-ask.modal.run",
    json={
        "question": "What are the opening hours?",
        "top_k": 5,
        "score_threshold": 0.30,
        "language_filter": None,
        "category_filter": None,
        "chat_history": "",
    }
)

print(response.json())
```

---

## Example Response

```json
{
  "question": "What are the opening hours?",
  "answer": "The museum is open daily from 9 AM...",
  "sources": [
    {
      "title": "Plan Your Visit",
      "url": "https://gem.eg/en/visit/plan-your-visit/",
      "category": "visitor_info",
      "language": "en",
      "score": 0.852
    }
  ],
  "query_type": "visitor_info",
  "query_language": "en",
  "chunks_retrieved": 5
}
```

---

# Evaluation

Run the evaluation suite:

```bash
python gem_evaluator.py
```

## Evaluation Metrics

| Metric | Description |
|---|---|
| Keyword Coverage | Expected keywords present in generated answers |
| Source Recall | Expected keywords present in retrieved sources |
| Language Correctness | Output language matches input language |
| Answer Completeness | Response length and completeness |
| Overall Score | Weighted evaluation score |

## Sample Results

```text
Questions Evaluated : 10
Average Score       : 0.87
Keyword Coverage    : 0.84
Source Recall       : 0.89
Language Accuracy   : 100%
```

---

# Known Limitations

| Limitation | Description | Mitigation |
|---|---|---|
| Limited knowledge base | 281 chunks may not cover all museum topics | Expand scraping coverage |
| Static knowledge | Data reflects a single scraping snapshot | Schedule periodic re-scraping |
| Cold start latency | First request after idle is slower | Keep one warm container active |
| Embedding limitations | Weak on negation and exact-number retrieval | Add hybrid BM25 retrieval |
| Rate limits | High query frequency may trigger provider limits | Use upgraded inference tiers |

---

# Future Improvements

| Improvement | Description |
|---|---|
| Hybrid Retrieval | Combine semantic retrieval with BM25 keyword search |
| Scheduled Re-Scraping | Automatically refresh museum content |
| Cross-Encoder Reranking | Improve retrieval precision |
| Larger Language Models | Improve Arabic fluency and reasoning |
| Multimodal Support | Add image-aware artifact retrieval |
| Expanded Dataset | Increase coverage across museum collections |

---

# Technical Notes

## Why Token-Based Chunking?

LLMs operate on tokens rather than characters. Token-aware chunking ensures consistent context windows across Arabic and English text.

## Why L2-Normalize Embeddings?

L2 normalization allows cosine similarity to be computed efficiently using inner products inside FAISS.

## Why Use query: and passage: Prefixes?

The multilingual-e5 model was trained using these prefixes and performs significantly worse without them.

## Why FAISS Instead of a Full Vector Database?

For smaller datasets, FAISS provides:
- simpler deployment,
- lower latency,
- exact similarity search,
- and no external infrastructure requirements.

---

# License

This project is licensed under the MIT License.

---

# Acknowledgements

- Grand Egyptian Museum
- Modal
- Groq
- Meta AI
- FAISS
- LangChain
- Intfloat
- Streamlit

---

<div align="center">

Built for bilingual museum question answering using Retrieval-Augmented Generation.

</div>
