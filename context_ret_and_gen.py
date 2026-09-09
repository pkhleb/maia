#!/usr/bin/env python3
"""
context_ret_and_gen.py — Context Retrieval and Generation techniques.

Section 4.1 of "A Survey of Context Engineering for Large Language Models"
(Mei et al. 2025, https://arxiv.org/pdf/2507.13334)

Techniques implemented:
    - self_refine: Generate → Critique → Refine (Madaan et al. 2023)

Planned:
    - multi_aspect_feedback
    - ...
"""
from model import call_model, ModelResult, MODEL
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from index_db import (
    init_db, get_or_create_index, get_file_checksum, upsert_file, 
    insert_nodes, get_all_nodes_with_embeddings,
    create_query_run, insert_chunk_scores,
)
import hashlib
import numpy as np
from sentence_transformers.util import cos_sim
from parser import parse_file
from pathlib import Path
DB_PATH = ".maia_index.db"

DEFAULT_INDEX = "flat-512"
DEFAULT_CONFIG = {"chunker": "fixed", "chunk_size": 512, "overlap": 64, "embedder": "all-MiniLM-L6-v2"}
DEFAULT_TOKEN_CONFIG = {"chunker": "fixed", "chunk_size": 128, "overlap": 96, "embedder": "all-MiniLM-L6-v2"}

init_db(DB_PATH)

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_tokenizer = AutoTokenizer.from_pretrained(MODEL)

# ── self-refine ───────────────────────────────────────────────────────────────

CRITIQUE_PROMPT = """Review the following response and identify specific problems or ways it could be improved.
Be concrete — point to exact issues rather than vague suggestions.
If the response is already good, say so explicitly.

Response to critique:
{response}"""

REFINE_PROMPT = """Revise the following response based on the critique provided.
Only change what the critique identifies as problematic. Do not introduce new issues.

Original response:
{response}

Critique:
{critique}

Revised response:"""


def self_refine(
    obs,
    reply: str,
    iterations: int = 1,
) -> tuple[str, list[ModelResult]]:
    """
    Refine a model response through generate → critique → refine iterations.

    Args:
        reply:      Initial model response to refine.
        iterations: Number of critique/refine cycles.
        obs:        Active observability flags.

    Returns:
        (final_reply, intermediate_results)
    """
    results: list[ModelResult] = []
    current = reply

    for i in range(iterations):
        # critique
        critique_result = call_model([
            {"role": "user", "content": CRITIQUE_PROMPT.format(response=current)}
        ])
        results.append(critique_result)
        critique = critique_result.reply

        obs("critique", "self_refine", f"iteration {i+1} critique:\n{critique}\n")

        # refine
        refine_result = call_model([
            {"role": "user", "content": REFINE_PROMPT.format(response=current, critique=critique)}
        ])
        results.append(refine_result)
        current = refine_result.reply

        obs("critique", "self_refine", f"iteration {i+1} refined:\n{current}\n")

    return current, results

def _score_all_chunks(query: dict, settings: dict, obs) -> tuple[int, list[dict]]:
    """
    Score every chunk in the index against the query and persist the run.
    Returns (query_run_id, [{"node_id", "score", "text", "metadata"}, ...]) for ALL chunks,
    not just top-k - required for token-level aggregation.
    """
    index_id = get_or_create_index(settings["name"], settings, DB_PATH)
    nodes = get_all_nodes_with_embeddings(index_id, DB_PATH)
    if not nodes:
        return None, []

    vectors = np.stack([n["embedding"] for n in nodes])
    query_vec = _embed_model.encode(query["content"])
    scores = cos_sim(query_vec, vectors)[0]

    query_run_id = create_query_run(index_id, query["content"], settings, DB_PATH)
    insert_chunk_scores(query_run_id, [
        {"node_id": n["id"], "score": float(scores[i])} for i, n in enumerate(nodes)
    ], DB_PATH)

    obs("retrieval", "token_rag", f"scored {len(nodes)} chunks -> query_run {query_run_id}")

    return query_run_id, [
        {"node_id": n["id"], "score": float(scores[i]), "text": n["text"], "metadata": n["metadata"]}
        for i, n in enumerate(nodes)
    ]

def _token_score(scored_chunks: list[dict], obs) -> dict[str, np.ndarray]:
    """
    Project chunk-level relevance scores onto a per-token axis, normalized by how
    many chunks cover each token. Scoped per file - token position is meaningless
    across files. Chunks without start_token/end_token (e.g AST chunks, for now)
    are skipped.

    Returns {file_path: token_score_array}.
    """
    by_file: dict[str, list[dict]] = {}
    for c in scored_chunks:
        meta = c["metadata"]
        if "start_token" not in meta or "end_token" not in meta:
            continue
        by_file.setdefault(meta["path"], []).append(c)

    heatmaps = {}
    for path, chunks in by_file.items():
        num_tokens = chunks[0]["metadata"]["num_tokens_in_file"]
        heat = np.zeros(num_tokens)
        coverage = np.zeros(num_tokens)
        for c in chunks:
            start, end = c["metadata"]["start_token"], c["metadata"]["end_token"]
            heat[start:end] += c["score"]
            coverage[start:end] += 1
        heatmaps[path] = heat / np.maximum(coverage, 1)

        obs("context", "token_rag",
            f"{path}: heatmap over {num_tokens} tokens, "
            f"max={heatmaps[path].max():.3f} min={heatmaps[path].min():.3f}")

    return heatmaps

def _fixed_chunk(file_path: str, settings: dict) -> list[dict]:
    text = Path(file_path).read_text(encoding="utf-8")
    token_ids = _tokenizer.encode(text, add_special_tokens=False)
    chunk_size, overlap = settings["chunk_size"], settings["overlap"]
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_ids = token_ids[start:start + chunk_size]
        chunks.append({
            "text": _tokenizer.decode(chunk_ids),
            "metadata": {
                "path": file_path, 
                "chunk_kind": "fixed",
                "start_token": start,
                "end_token": end,
                "num_tokens_in_file": len(token_ids),
            },
        })
        start += chunk_size - overlap
    return chunks

def _ast_chunk(file_path: str, settings: dict) -> list[dict]:
    result = parser.parse_file(file_path)
    return [
        {
            "text": sym.source,
            "metadata": {
                "path": file_path,
                "chunk_kind": sym.kind,
                "qualified_name": sym.qualified_name,
                "start_line": sym.start_line,
                "end_line": sym.end_line,
                "docstring": sym.docstring,
            },
        }
        for sym in result.symbols
    ]

CHUNKERS = {"fixed": _fixed_chunk, "ast": _ast_chunk}

def _index(file_paths: list[str], settings: dict) -> None:
    index_id = get_or_create_index(settings["name"], settings, DB_PATH)
    chunker = CHUNKERS.get(settings["chunker"])
    if chunker is None:
        raise ValueError(f"unknown chunker: {settings['chunker']}")

    for path in file_paths:
        checksum = hashlib.sha256(Path(path).read_bytes()).hexdigest()

        if get_file_checksum(index_id, path, DB_PATH) == checksum:
            continue # unchanged, skip
        
        file_id = upsert_file(index_id, path, checksum, DB_PATH)
        chunks = chunker(path, settings)
        if not chunks:
            continue
        embeddings = _embed_model.encode([c["text"] for c in chunks])

        insert_nodes(index_id, [
            {"text": c["text"], "embedding": vec, "file_id": file_id, "metadata": c["metadata"]}
            for c, vec in zip(chunks, embeddings)
        ], DB_PATH)

def _retrieve(query: dict, settings: dict, obs: set[str] = set()) -> list[str]:
    index_id = get_or_create_index(settings["name"], settings, DB_PATH)
    nodes = get_all_nodes_with_embeddings(index_id, DB_PATH)

    if not nodes:
        return []
    
    vectors = np.stack([n["embedding"] for n in nodes])
    query_vec = _embed_model.encode(query["content"])
    scores = cos_sim(query_vec, vectors)[0]
    top_indices = scores.argsort(descending=True)[:settings["top_k"]]

    obs("retrieval", "rag", f"retrieved {settings['top_k']} chunks:")
    for rank, i in enumerate(top_indices):
        obs("retrieval", "rag", f"  [{rank+1}] score={scores[i]:.3f} | {nodes[i]['text'][:80].strip()!r}")
    obs("retrieval", "rag", "\n")

    return [nodes[i]["text"] for i in top_indices]

def _assemble(chunks: list[str], obs) -> str:
    output = "\n\n".join(chunks)

    obs("context", "rag", f"assembled context ({len(output)} chars):\n{output}\n")

    return output

def rag(file_paths: list[str], query: dict, settings: dict, obs) -> str:
    _index(file_paths, settings)
    chunks = _retrieve(query, settings, obs)
    return _assemble(chunks, obs)

def token_rag(file_paths: list[str], query: dict, settings: dict, obs) -> str:
    _index(file_paths, settings)
    _, scored_chunks = _score_all_chunks(query, settings, obs)
    return _token_score(scored_chunks, obs)

def peak_span(file_path, heat, window=64):
    from pathlib import Path
    token_ids = _tokenizer.encode(Path(file_path).read_text(encoding="utf-8"), add_special_tokens=False)
    best_start, best_score = 0, float("-inf")
    for start in range(0, max(len(heat) - window, 0) + 1):
        s = heat[start:start + window].sum()
        if s > best_score:
            best_start, best_score = start, s
    return _tokenizer.decode(token_ids[best_start:best_start + window])
