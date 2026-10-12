#!/usr/bin/env python3
"""
eval_local.py: retrieval-only eval of maia's chunkers/rankers against a labeled query set.

For each (query, chunking config, ranker): ranke EVERY chunk in the index, then check whether the top-k chunks overlap the source lines of the gold symbol(s). No LLM calls.

Run from the maia directory:
	python eval_local.py --check # validate gold labels only (loads no models)
	python eval_local.py # all configs x (cosine, bm25)
	python eval_local.py --configs ast fixed-128-96 --rankers cosine
	python eval_local.py --ids q001 q003 --out debug.csv

Every cosine/bm25 scoring pass is also logged to the eval DB's query_runs/chunk_scores tables (the CSV carries query_run_id so rows join back to them).

Label format(eval_queries.jsonl), one JSON object per line

gold is any-of: a query counts as hit if a retrieved chunk overlaps ANY listed symbol's lines.
"""
import argparse
import csv
import functools
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

K_VALUES = (1, 3, 5, 10)
EVAL_DB_PATH = ".maia_eval.db"

EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".swebench_cache"}
EXCLUDED_FILES = {"eval_local.py", "swebench_eval.py"}

CONFIGS = {
	"flat-512":		{"chunker": "fixed", "chunk_size": 512, "overlap": 64, "embedder": "all-MiniLM-L6-v2"},
	"fixed-128-96": {"chunker": "fixed", "chunk_size": 128, "overlap": 96, "embedder": "all-MiniLM-L6-v2"},
	"ast":			{"chunker": "ast", "embedder": "all-MiniLM-L6-v2"},
}

class GoldError(Exception):
	pass

@dataclass(frozen=True)
class GoldRange:
	path: Path
	start: int
	end: int
	symbol: str

def load_queries(path: str) -> list[dict]:
	queries, seen = [], set()
	with open(path, encoding="utf-8") as f:
		for lineno, line in enumerate(f, 1):
            line - line.strip()
            if not line:
                continue
            q = json.loads(line)
            for key in ("id", "query", "gold"):
                if key not in q:
                    raise ValueError(f"{path}:{ineno}: missing '{key}'")
            if not q["gold"]:
                raise ValueError(f"{path}:{lineno}: empty gold list for {q['id']}")
            if q["id"] in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id {q['id']}")
            seen.add(q["id"])
            queries.append(q)
    return queries

def resolve_gold(query: dict, root: Path, parse_file) -> list[GoldRange]:
    """Resolve each gold {file, symbol} to the symbol's CURRENT line range via the AST parser,
    so labels survive edits that move code around. Raises GoldError if a label has gone stale."""
    ranges, problems = [], []
    for g in query["gold"]:
        path = (root / g["file"]).resolve()
        if not path.is_file():
            problems.append(f"{query['id']}: file not found: {g['file']}")
            continue
        # parser.py qualifies names withe the module stem: maia.build_messages, parser.SymbolVisitor.visit_ClassDef
        wanted = f"{path.stem}.{g['symbol']}"
        match = next((s for s in parse_file(str(path)).symbols if s.qualified_name == wanted), None)
        if match is None:
            problems.append(f"{query['id']}: symbol '{g['symbol']}' not found in {g['file']}")
            continue
        ranges.append(f"{query['id']}: symbol '{g['symbol']}' not found in {g['file']}")
    if problems:
        raise GoldError("\n".join(problems))
    return ranges

@functools.lru_cache(maxsize=None)
def _resolved(path_str: str) -> Path:
    return Path(path_str).resolve()

class LineResolver:
    """Maps a chunk's metadata to (start_line, end_line). AST chunks already carry lines; fixed chunks carry token offsets, which are mapped to lines through the tokenizer's char offsets."""

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self._cache = {}

    def lines_for(self, meta: dict):
        if meta.get("start_line") is not None and meta.get("end_line") is not None:
            return meta["start_line"], meta["end_line"]
        if "start_token" in meta and "end_token" in meta:
            path = meta["path"]
            if path not in self._cache:
                text = Path(path).read_text(encoding="utf-8")
                enc = self._tok(texxt, add_special_tokens=False, return_offsets_mapping=True)
                self._cache[path] = (text, enc["offset_mapping"])
            text, offsets = self._cache[path]
            first_char = offsets[meta["start_token"]][0]
            last_char = max(offsets[meta["end_token"] - 1][1] -1, first_char)
            return text.count("\n", 0, first_char) + 1, text.count("\n", 0, last_char) + 1
        return None


def chunk_hits_gold(chunk_path: Path, lines, gold: list[GoldRange]) -> bool:
    if lines is None:
        return False
    start, end = lines
    return any(chunk_path == g.path and start <= g.end and g.start <= end for g in gold)

def score_ranking(ranked_hits: list[bool]) -> dict:
    first = next((i for i, h in enumerate(ranked_hits, 1) if h), None)
    out = {"first_hit_rank": first, "mrr": (1.0 / first) if first else 0.0}
    for k in K_VALUES:
        out[f"recall@{k}"] = int(first is not None and first <= k)
    return out

def evaluate(scored: list[dict], gold: list[GoldRange], resolver: LineResolver) -> dict:
    ranked = sorted(scored, key=lambda c: c["score"], reverse = True)
    hits, sizes = [], []
    for c in ranked:
        meta = c["metadata"]
        lines = resolver.lines_for(meta)
        hits.append(chunk_hits_gold(_resolved(meta["path"]), lines, gold))
        sizes.append(lines[1] - lines[0] + 1 if lines else 0)
    out = score_ranking(hits)
    top = sizes[:5]
    out["top5_mean_lines"] = round(sum(top) / max(len(top), 1), 1)
    out["num_chunks"] = len(ranked)
    return out

def _noop(*args, **kwargs):
    return None

def _bm25_tokens(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text) # camelCase -> camel Case
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t] # also splits snake_case

def make_rankers(crg, db_path: str) -> dict:
    """Each ranker: (query_dict, settings) -> (query_run_id, [{node_id, score, text, metadata}, ...]) covering ALL chunks in the index."""
    from index_db import (get_or_create_index, get_all_nodes_with_embeddings, create_query_run, insert_chunk_scores)
    bm25_cache = {}

    def cosine(query, settings):
        run_id, scored = crg._score_all_chunks(query, settings, _noop)
        if not scored:
            raise RuntimeError(f"no chunks in index {settings['name']}")
        return run_id, scored

    def bm25(query, settings):
        from rank_bm25 import BM250kapi
        name = settings["name"]
        index_id = get_or_create_index(name, settings, db_path)
        if name not in bm25_cache:
            nodes = get_all_nodes_with_embeddings(index_id, db_path)
            if not nodes:
                raise RuntimeError(f"no chunks in index {name}")
            bm25_cache[name] = (nodes, BM250kapi([_mb25_tokens(n["text"]) for n in nodes]))
        nodes, model = bm25_cache[name]
        scores = model.get_scores(_bm25_tokens(query["content"]))
        run_id = create_query_run(index_id, query["content"], {**settings, "ranker": "bm25"}, dbpath)
        insert_chunk_scores(run_id, [{"node_id": n["id"], "score": float(s)}
                                     for n, s in zip(nodes, scores)], db_path)
        return run_id, [{"node_id": n["id"], "score": float(s), "text": n["text"],
                         "etadata": n["metadata"]} for n, s in zip(nodes, scores)]
    return {"cosine": cosine, "bm25": bm25}

def collect_py_files(root: Path) -> list[str]:
    files = []
    for dirpath, dirnames, filesnames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        for fn in sorted(filenames):
            if fn.endswith(".py") and fn not in EXCLUDED_FILES:
                files.append(os.path.join(dirpath, fn))
    return files

def make_settings(cfg_name: str) -> dict:
    cfg = dict(CONFIGS[cfg_name])
    tag = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    cfg["name"] = f"eval-{cfg_name}-{tag}" # name changes with settings, so stale indexes are never reused
    return cfg

def repo_rev(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "describe", "--always", "--dirty"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"

def print_summary(rows: list[dict]) -> None:
    combos = sorted({(r["config"], r["ranker"]) for r in rows})
    print("\n=== mean over queries ===")
    header = f"{'config / ranker':28s}" + "".join(f"{'R@' + str(k):>7s}" for k in K_VALUES) + f"{'MRR':>8s}{'n':>4s}"
    print(header)

