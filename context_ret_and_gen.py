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
from index_db import init_db, get_or_create_index, get_file_checksum, upsert_file, insert_nodes, get_all_nodes_with_embeddings
import hashlib
import numpy as np
from sentence_transformers.util import cos_sim
DB_PATH = ".maia_index.db"

DEFAULT_INDEX = "flat-512"
DEFAULT_CONFIG = {"chunker": "fixed", "chunk_size": 512, "overlap": 64, "embedder": "all-MiniLM-L6-v2"}

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

def _fixed_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    token_ids = _tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(token_ids):
        chunk_ids = token_ids[start:start + chunk_size]
        chunks.append(_tokenizer.decode(chunk_ids))
        start += chunk_size - overlap
    return chunks

def _index(files: dict[str, str], settings: dict) -> None:
    index_id = get_or_create_index(settings["name"], settings, DB_PATH)

    for path, text in files.items():
        checksum = hashlib.sha256(text.encode()).hexdigest()

        if get_file_checksum(index_id, path, DB_PATH) == checksum:
            continue # unchanged, skip
        
        file_id = upsert_file(index_id, path, checksum, DB_PATH)
        if settings["chunker"] == "fixed":
            chunks = _fixed_chunk(text, settings["chunk_size"], settings["overlap"])
        else:
            raise ValueError(f"unknown chunker: {settings['chunker']}")
        embeddings = _embed_model.encode(chunks)

        insert_nodes(index_id, [
            {"text": chunk, "embedding": vec, "file_id": file_id, "metadata": {"path": path}}
            for chunk, vec in zip(chunks, embeddings)
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

def rag(files: dict, query: dict, settings: dict, obs) -> str:
    _index(files, settings)
    chunks = _retrieve(query, settings, obs)
    return _assemble(chunks, obs)

