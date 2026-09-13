# MAIA

**MAIA** is an experimental coding assistant and ML systems project for evaluating repository-level retrieval, context construction, and LLM inference strategies.

The project focuses on the infrastructure around LLM applications: **indexing, retrieval, experiment tracking, observability, reproducibility, and inference pipelines**.

## Features

* Incremental repository indexing using file hashes to avoid unnecessary re-embedding
* Semantic retrieval with sentence-transformer embeddings and cosine similarity
* Multiple context-generation strategies, including fixed-size, AST-aware, and token-level relevance retrieval
* Persistent SQLite storage for indexed artifacts, embeddings, retrieval scores, and query runs
* Configurable inference through OpenAI-compatible model endpoints
* Generate → critique → refine inference workflow
* Retrieval and token-usage observability for debugging and evaluation

## Architecture

```text
Repository
    ↓
Incremental Indexing
    ↓
Embeddings + Metadata
    ↓
Retrieval Strategy
    ↓
Context Construction
    ↓
LLM Inference
    ↓
Telemetry / Experiment Data
```

MAIA is designed as a testbed for questions such as:

* How do different code-chunking strategies affect retrieval quality?
* What is the tradeoff between retrieval accuracy, context size, latency, and token usage?
* How can retrieval configurations and inference runs be made reproducible and observable?

## Setup

```bash
git clone https://github.com/pkhleb/maia.git
cd maia

pip install -r requirements.txt
```

Configure the OpenAI-compatible endpoint and model credentials in your environment, then run:

```bash
python maia.py
```

## Current Development

Current work is focused on expanding MAIA into a reproducible evaluation platform for repository-level retrieval, including automated retrieval benchmarks, experiment/version tracking, structured observability, testing, and CI/CD.
