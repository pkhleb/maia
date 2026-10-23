#!/usr/bin/env python3
"""
eval_local.py — retrieval-only eval of maia's chunkers/rankers against a labeled query set.

For each (query, chunking config, ranker): rank EVERY chunk in the index, then check whether
the top-k chunks overlap the source lines of the gold symbol(s). No LLM calls.

Run from the maia directory:
    python eval_local.py --check                       # validate gold labels only (loads no models)
    python eval_local.py                               # all configs x (cosine, bm25)
    python eval_local.py --configs ast fixed-128-96 --rankers cosine
    python eval_local.py --ids q001 q003 --out debug.csv

Every cosine/bm25 scoring pass is also logged to the eval DB's query_runs/chunk_scores tables
(the CSV carries query_run_id so rows join back to them).

Label format (eval_queries.jsonl), one JSON object per line:
    {"id": "q001", "query": "...", "gold": [{"file": "maia.py", "symbol": "build_messages"}], "note": "..."}
`gold` is any-of: a query counts as hit if a retrieved chunk overlaps ANY listed symbol's lines.
"""
import argparse
import bisect
import csv
import functools
import hashlib
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

K_VALUES = (1, 3, 5, 10)
BUDGETS = (256, 512, 1024, 2048)    # context-token budgets for the equal-budget metric
DEDUPE_SWEEP = ("off", "0.8", "0.5", "0.3", "0.0")   # max fraction of a chunk allowed to repeat kept text
# paired comparisons "A:B" (system = config/ranker) reported with bootstrap CIs; A - B is shown
DEFAULT_COMPARISONS = (
    "ast/cosine:fixed-128-96/cosine",
    "ast/cosine:flat-512/bm25",
    "ast/cosine:ast/bm25",
    "fixed-128-96/cosine:fixed-128-96/bm25",
    "fixed-128-96/cosine:flat-512/bm25",
)
COV_THRESHOLD = 0.8                # a gold symbol counts as covered when >= this fraction of its tokens are in context
HEADLINE_DEDUPE = "0.5"           # fixed in advance; the full tables are printed at this setting
EVAL_DB_PATH = ".maia_eval.db"      # isolated from .maia_index.db, which maia.py chats against

# directories / files never indexed for eval (the .swebench_cache dir holds full astropy/django checkouts)
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".swebench_cache"}
EXCLUDED_FILES = {"eval_local.py", "swebench_eval.py"}

CONFIGS = {
    "flat-512":     {"chunker": "fixed", "chunk_size": 512, "overlap": 64, "embedder": "all-MiniLM-L6-v2"},
    "fixed-128-96": {"chunker": "fixed", "chunk_size": 128, "overlap": 96, "embedder": "all-MiniLM-L6-v2"},
    "ast":          {"chunker": "ast", "embedder": "all-MiniLM-L6-v2"},
}


# ── labels ───────────────────────────────────────────────────────────────────

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
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            for key in ("id", "query", "gold"):
                if key not in q:
                    raise ValueError(f"{path}:{lineno}: missing '{key}'")
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
        # parser.py qualifies names with the module stem: maia.build_messages, parser.SymbolVisitor.visit_ClassDef
        wanted = f"{path.stem}.{g['symbol']}"
        match = next((s for s in parse_file(str(path)).symbols if s.qualified_name == wanted), None)
        if match is None:
            problems.append(f"{query['id']}: symbol '{g['symbol']}' not found in {g['file']}")
            continue
        ranges.append(GoldRange(path, match.start_line, match.end_line, g["symbol"]))
    if problems:
        raise GoldError("\n".join(problems))
    return ranges


# ── chunk -> line range ──────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _resolved(path_str: str) -> Path:
    return Path(path_str).resolve()


class LineResolver:
    """Maps a chunk's metadata to (start_line, end_line). AST chunks already carry lines; fixed
    chunks carry token offsets, which are mapped to lines through the tokenizer's char offsets."""

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self._cache = {}
        self._costs = {}

    def _load(self, path: str):
        if path not in self._cache:
            text = Path(path).read_text(encoding="utf-8")
            enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
            self._cache[path] = (text, enc["offset_mapping"])
        return self._cache[path]

    def line_costs(self, path: str) -> list[int]:
        """Tokens per source line (index 0 = line 1), counted with the SAME tokenizer for every
        chunker. A token belongs to the line its first character is on. This is the common
        currency for the token budget: any chunk, however it was cut, costs the sum of its lines."""
        if path not in self._costs:
            text, offsets = self._load(path)
            line_starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n"]
            costs = [0] * len(line_starts)
            for start, _end in offsets:
                costs[bisect.bisect_right(line_starts, start) - 1] += 1
            self._costs[path] = costs
        return self._costs[path]

    def lines_for(self, meta: dict):
        if meta.get("start_line") is not None and meta.get("end_line") is not None:
            return meta["start_line"], meta["end_line"]
        if "start_token" in meta and "end_token" in meta:
            path = meta["path"]
            text, offsets = self._load(path)
            first_char = offsets[meta["start_token"]][0]
            last_char = max(offsets[meta["end_token"] - 1][1] - 1, first_char)
            return text.count("\n", 0, first_char) + 1, text.count("\n", 0, last_char) + 1
        return None


# ── metrics ──────────────────────────────────────────────────────────────────

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


def parse_threshold(label: str):
    """'off' -> None (no de-duplication); otherwise a float max-overlap fraction."""
    return None if label == "off" else float(label)


def dedupe_ranked(ranked: list[dict], resolver: LineResolver, max_overlap: float) -> list[dict]:
    """Greedy non-maximum suppression. Walk the ranking best-first and drop any chunk whose lines
    are more than `max_overlap` already covered by chunks we kept. Order is preserved."""
    covered: dict[Path, set] = {}
    kept = []
    for c in ranked:
        meta = c["metadata"]
        lines = resolver.lines_for(meta)
        if lines is None:                      # no line info: can't judge, keep it
            kept.append(c)
            continue
        span = set(range(lines[0], lines[1] + 1))
        seen = covered.setdefault(_resolved(meta["path"]), set())
        if len(span & seen) / len(span) > max_overlap:
            continue                           # mostly a repeat of something ranked higher
        kept.append(c)
        seen |= span
    return kept


def _gold_token_info(gold: list[GoldRange], resolver: LineResolver) -> list[tuple]:
    """(gold range, tokens-per-line, total tokens in the symbol) for each gold symbol."""
    info = []
    for g in gold:
        costs = resolver.line_costs(str(g.path))
        total = sum(costs[ln - 1] for ln in range(g.start, g.end + 1) if ln - 1 < len(costs))
        if total > 0:
            info.append((g, costs, total))
    return info


def _coverage(covered: dict, gold_info: list[tuple]) -> float:
    """Fraction of the best-covered gold symbol's tokens that are in the context. Gold is any-of,
    so the query is scored by whichever listed symbol is covered most."""
    best = 0.0
    for g, costs, total in gold_info:
        seen = covered.get(g.path)
        if not seen:
            continue
        got = sum(costs[ln - 1] for ln in range(g.start, g.end + 1) if ln in seen and ln - 1 < len(costs))
        best = max(best, got / total)
    return best


def evaluate(scored: list[dict], gold: list[GoldRange], resolver: LineResolver,
             budgets=BUDGETS, dedupe=None, cov_threshold=COV_THRESHOLD) -> dict:
    """`dedupe` is None (off) or a max-overlap fraction; see dedupe_ranked.
    `cov_threshold` is the fraction of the gold symbol that must be in the context to count as covered."""
    ranked = sorted(scored, key=lambda c: c["score"], reverse=True)
    n_raw = len(ranked)
    if dedupe is not None:
        ranked = dedupe_ranked(ranked, resolver, dedupe)
    hits, sizes = [], []
    # Token budget: walk the ranking the way a context assembler would, adding chunks until the
    # budget runs out. Cost is the number of NEW tokens a chunk adds (lines already covered by
    # an earlier, overlapping chunk are free), because the duplicate text is merged, not sent twice.
    # Gold coverage: the same walk also tracks how much of each gold symbol (in tokens) is inside
    # the context built so far, and snapshots it for every budget (the last chunk that still fits).
    gold_info = _gold_token_info(gold, resolver)
    cov_at = {b: 0.0 for b in budgets}
    max_budget = max(budgets)
    covered: dict[Path, set] = {}
    cum, tokens_at_hit = 0, None
    for c in ranked:
        meta = c["metadata"]
        lines = resolver.lines_for(meta)
        path = _resolved(meta["path"])
        hit = chunk_hits_gold(path, lines, gold)
        hits.append(hit)
        sizes.append(lines[1] - lines[0] + 1 if lines else 0)
        if lines and (tokens_at_hit is None or cum <= max_budget):
            seen = covered.setdefault(path, set())
            costs = resolver.line_costs(meta["path"])
            new = [ln for ln in range(lines[0], lines[1] + 1) if ln not in seen]
            cum += sum(costs[ln - 1] for ln in new if ln - 1 < len(costs))
            seen.update(new)
            if hit and tokens_at_hit is None:
                tokens_at_hit = cum
            if cum <= max_budget:
                cov = _coverage(covered, gold_info)
                for b in budgets:
                    if cum <= b:
                        cov_at[b] = cov
    out = score_ranking(hits)
    out["tokens_to_first_hit"] = tokens_at_hit
    for b in budgets:
        out[f"budget@{b}"] = int(tokens_at_hit is not None and tokens_at_hit <= b)
        out[f"gold_cov@{b}"] = round(cov_at[b], 4)
        out[f"cov80@{b}"] = int(cov_at[b] + 1e-9 >= cov_threshold)
    top = sizes[:5]
    out["top5_mean_lines"] = round(sum(top) / max(len(top), 1), 1)   # rough context-size cost
    out["num_chunks"] = n_raw
    out["num_chunks_deduped"] = len(ranked)
    return out


# ── rankers ──────────────────────────────────────────────────────────────────

def _noop(*args, **kwargs):
    return None


def _bm25_tokens(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)          # camelCase -> camel Case
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]   # also splits snake_case


def make_rankers(crg, db_path: str) -> dict:
    """Each ranker: (query_dict, settings) -> (query_run_id, [{node_id, score, text, metadata}, ...])
    covering ALL chunks in the index."""
    from index_db import (get_or_create_index, get_all_nodes_with_embeddings,
                          create_query_run, insert_chunk_scores)
    bm25_cache = {}

    def cosine(query, settings):
        run_id, scored = crg._score_all_chunks(query, settings, _noop)
        if not scored:
            raise RuntimeError(f"no chunks in index {settings['name']}")
        return run_id, scored

    def bm25(query, settings):
        from rank_bm25 import BM25Okapi
        name = settings["name"]
        index_id = get_or_create_index(name, settings, db_path)
        if name not in bm25_cache:
            nodes = get_all_nodes_with_embeddings(index_id, db_path)
            if not nodes:
                raise RuntimeError(f"no chunks in index {name}")
            bm25_cache[name] = (nodes, BM25Okapi([_bm25_tokens(n["text"]) for n in nodes]))
        nodes, model = bm25_cache[name]
        scores = model.get_scores(_bm25_tokens(query["content"]))
        run_id = create_query_run(index_id, query["content"], {**settings, "ranker": "bm25"}, db_path)
        insert_chunk_scores(run_id, [{"node_id": n["id"], "score": float(s)}
                                     for n, s in zip(nodes, scores)], db_path)
        return run_id, [{"node_id": n["id"], "score": float(s), "text": n["text"],
                         "metadata": n["metadata"]} for n, s in zip(nodes, scores)]

    return {"cosine": cosine, "bm25": bm25}


# ── plumbing ─────────────────────────────────────────────────────────────────

def collect_py_files(root: Path) -> list[str]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        for fn in sorted(filenames):
            if fn.endswith(".py") and fn not in EXCLUDED_FILES:
                files.append(os.path.join(dirpath, fn))
    return files


def make_settings(cfg_name: str) -> dict:
    cfg = dict(CONFIGS[cfg_name])
    tag = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    cfg["name"] = f"eval-{cfg_name}-{tag}"   # name changes with settings, so stale indexes are never reused
    return cfg


def repo_rev(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "describe", "--always", "--dirty"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def print_budget_summary(rows: list[dict], budgets) -> None:
    combos = sorted({(r["config"], r["ranker"]) for r in rows})
    print("\n=== equal token budget: fraction of queries whose gold is in the context ===")
    print("(chunks added in rank order; overlapping text counted once; never reached = miss)")
    header = f"{'config / ranker':28s}" + "".join(f"{'T<=' + str(b):>8s}" for b in budgets) + \
             f"{'median tok':>12s}{'n':>4s}"
    print(header)
    for cfg, ranker in combos:
        sub = [r for r in rows if r["config"] == cfg and r["ranker"] == ranker]
        line = f"{cfg + ' / ' + ranker:28s}"
        for b in budgets:
            line += f"{sum(r[f'budget@{b}'] for r in sub) / len(sub):8.2f}"
        # misses count as infinity, so the median is only finite once more than half the queries hit
        toks = sorted(r["tokens_to_first_hit"] if r["tokens_to_first_hit"] is not None else float("inf")
                      for r in sub)
        mid = toks[len(toks) // 2] if len(toks) % 2 else (toks[len(toks) // 2 - 1] + toks[len(toks) // 2]) / 2
        line += f"{'inf' if mid == float('inf') else f'{mid:.0f}':>12s}{len(sub):4d}"
        print(line)


def print_coverage_summary(rows: list[dict], budgets, cov_threshold: float) -> None:
    combos = sorted({(r["config"], r["ranker"]) for r in rows})
    print("\n=== gold coverage: share of the gold symbol's tokens inside the context ===")
    print("(chunks added in rank order within the budget; overlap counted once; any-of gold scored by its "
          "best-covered symbol)")
    for title, key, fmt in [("mean coverage", "gold_cov", ".2f"),
                            (f"fraction of queries with >= {cov_threshold:.0%} covered", "cov80", ".2f")]:
        print(f"\n{title}")
        print(f"{'config / ranker':28s}" + "".join(f"{'T<=' + str(b):>8s}" for b in budgets) + f"{'n':>4s}")
        for cfg, ranker in combos:
            sub = [r for r in rows if r["config"] == cfg and r["ranker"] == ranker]
            line = f"{cfg + ' / ' + ranker:28s}"
            for b in budgets:
                line += f"{sum(r[f'{key}@{b}'] for r in sub) / len(sub):{8}{fmt}}"
            print(line + f"{len(sub):4d}")


def _sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test on the non-tied queries."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def print_bootstrap(rows: list[dict], budgets, comparisons, n_boot: int = 10000, seed: int = 0) -> None:
    """Paired bootstrap over queries. For each A:B pair and metric, take the per-query difference
    A - B, resample queries with replacement, and report the mean difference with a 95% percentile CI.
    The same resamples are reused for every metric of a pair, so the metrics stay paired."""
    import numpy as np
    by_sys: dict[str, dict] = {}
    for r in rows:
        by_sys.setdefault(f"{r['config']}/{r['ranker']}", {})[r["id"]] = r
    metrics = [("MRR", "mrr"), ("R@3", "recall@3")] + [(f"coverage@{b}", f"gold_cov@{b}") for b in budgets]
    rng = np.random.default_rng(seed)
    resamples: dict[int, "np.ndarray"] = {}
    print(f"\n=== paired bootstrap: A - B per query ({n_boot} resamples, 95% percentile CI) ===")
    print("W/L/T = queries where A beats / loses to / ties B.  * = CI excludes 0.  p = exact sign test.")
    tests = 0
    for comp in comparisons:
        a_name, _, b_name = comp.partition(":")
        if a_name not in by_sys or b_name not in by_sys:
            print(f"\n{a_name}  vs  {b_name}: skipped (system not in this run)")
            continue
        ids = sorted(set(by_sys[a_name]) & set(by_sys[b_name]))
        n = len(ids)
        if n not in resamples:
            resamples[n] = rng.integers(0, n, size=(n_boot, n))
        idx = resamples[n]
        print(f"\n{a_name}  vs  {b_name}   (n={n})")
        print(f"  {'metric':14s}{'A - B':>9s}   {'95% CI':^17s}{'W/L/T':>10s}{'p':>8s}")
        for title, key in metrics:
            d = np.array([by_sys[a_name][i][key] - by_sys[b_name][i][key] for i in ids], dtype=float)
            lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
            w, l = int((d > 0).sum()), int((d < 0).sum())
            star = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {title:14s}{d.mean():+9.3f}   [{lo:+7.3f},{hi:+7.3f}]{star}"
                  f"{f'{w}/{l}/{n - w - l}':>10s}{_sign_test_p(w, l):8.3f}")
            tests += 1
    print(f"\n{tests} comparisons: about {tests / 20:.1f} would look significant by chance alone, "
          "so treat starred rows as leads to confirm, not findings.")


def print_source_split(rows: list[dict], budgets) -> None:
    """Means per query origin (e.g. 'generated' vs 'chat'). No CIs: the groups are only 10-13 queries."""
    sources = sorted({r["source"] for r in rows})
    if len(sources) < 2:
        return
    combos = sorted({(r["config"], r["ranker"]) for r in rows})
    for s in sources:
        sub_rows = [r for r in rows if r["source"] == s]
        nq = len({r["id"] for r in sub_rows})
        print(f"\n=== source = {s} ({nq} queries): MRR, R@3, mean gold coverage per budget ===")
        print(f"{'config / ranker':28s}{'MRR':>7s}{'R@3':>6s}" + "".join(f"{'cov@' + str(b):>9s}" for b in budgets))
        for cfg, ranker in combos:
            sub = [r for r in sub_rows if r["config"] == cfg and r["ranker"] == ranker]
            line = f"{cfg + ' / ' + ranker:28s}{sum(r['mrr'] for r in sub) / len(sub):7.3f}"
            line += f"{sum(r['recall@3'] for r in sub) / len(sub):6.2f}"
            for b in budgets:
                line += f"{sum(r[f'gold_cov@{b}'] for r in sub) / len(sub):9.2f}"
            print(line)


def print_summary(rows: list[dict]) -> None:
    combos = sorted({(r["config"], r["ranker"]) for r in rows})
    print("\n=== mean over queries ===")
    header = f"{'config / ranker':28s}" + "".join(f"{'R@' + str(k):>7s}" for k in K_VALUES) + f"{'MRR':>8s}{'n':>4s}"
    print(header)
    for cfg, ranker in combos:
        sub = [r for r in rows if r["config"] == cfg and r["ranker"] == ranker]
        line = f"{cfg + ' / ' + ranker:28s}"
        for k in K_VALUES:
            line += f"{sum(r[f'recall@{k}'] for r in sub) / len(sub):7.2f}"
        line += f"{sum(r['mrr'] for r in sub) / len(sub):8.3f}{len(sub):4d}"
        print(line)

    print("\n=== rank of first hit per query ('-' = never retrieved) ===")
    cols = [f"{c}/{r}" for c, r in combos]
    print(f"{'query':8s}" + "".join(f"{c:>{max(len(c), 8) + 2}s}" for c in cols))
    for qid in dict.fromkeys(r["id"] for r in rows):
        line = f"{qid:8s}"
        for (cfg, ranker), col in zip(combos, cols):
            match = next((r for r in rows if r["id"] == qid and r["config"] == cfg and r["ranker"] == ranker), None)
            cell = "" if match is None else ("-" if match["first_hit_rank"] is None else str(match["first_hit_rank"]))
            line += f"{cell:>{max(len(col), 8) + 2}s}"
        print(line)


def print_sweep(rows: list[dict], thresholds: list[str], budgets) -> None:
    """One pivot per metric: systems down the side, de-dup thresholds across the top."""
    combos = sorted({(r["config"], r["ranker"]) for r in rows})
    metrics = [("MRR", "mrr", ".3f"), ("R@3", "recall@3", ".2f")]
    metrics += [(f"fraction hit within {b} tokens", f"budget@{b}", ".2f") for b in budgets]
    metrics += [(f"mean gold coverage within {b} tokens", f"gold_cov@{b}", ".2f") for b in budgets]
    print("\n=== de-duplication sweep (column = max fraction of a chunk allowed to repeat kept text) ===")
    for title, key, fmt in metrics:
        print(f"\n{title}")
        print(f"{'config / ranker':28s}" + "".join(f"{t:>8s}" for t in thresholds))
        for cfg, ranker in combos:
            line = f"{cfg + ' / ' + ranker:28s}"
            for t in thresholds:
                sub = [r for r in rows if r["config"] == cfg and r["ranker"] == ranker and r["dedupe"] == t]
                line += f"{sum(r[key] for r in sub) / len(sub):{8}{fmt}}" if sub else f"{'':>8s}"
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default="eval_queries.jsonl")
    ap.add_argument("--root", default=".", help="repo to index and evaluate against (default: cwd)")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--rankers", nargs="+", default=["cosine", "bm25"], choices=["cosine", "bm25"])
    ap.add_argument("--ids", nargs="+", default=None, help="only run these query ids")
    ap.add_argument("--out", default="eval_results.csv")
    ap.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS),
                    help="context-token budgets for the equal-budget metric")
    ap.add_argument("--compare", nargs="+", default=list(DEFAULT_COMPARISONS), metavar="A:B",
                    help="paired comparisons for the bootstrap, as 'config/ranker:config/ranker'")
    ap.add_argument("--boot-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cov-threshold", type=float, default=COV_THRESHOLD,
                    help="fraction of the gold symbol that must be in context to count as covered")
    ap.add_argument("--dedupe", nargs="+", default=list(DEDUPE_SWEEP), metavar="T",
                    help="de-dup settings to evaluate: 'off' or a max-overlap fraction in [0,1] "
                         f"(default: {' '.join(DEDUPE_SWEEP)})")
    ap.add_argument("--check", action="store_true", help="only validate that every gold symbol resolves")
    ap.add_argument("--fresh", action="store_true",
                    help="delete each eval index first (also deletes its query_runs history)")
    args = ap.parse_args()
    try:
        for label in args.dedupe:
            parse_threshold(label)
    except ValueError:
        raise SystemExit(f"--dedupe values must be 'off' or numbers, got {args.dedupe}")
    headline = HEADLINE_DEDUPE if HEADLINE_DEDUPE in args.dedupe else args.dedupe[0]

    root = Path(args.root).resolve()
    queries = load_queries(args.queries)
    if args.ids:
        queries = [q for q in queries if q["id"] in set(args.ids)]
        if not queries:
            raise SystemExit(f"no queries matched --ids {args.ids}")

    from parser import parse_file
    cached_parse = functools.lru_cache(maxsize=None)(parse_file)
    gold, problems = {}, []
    for q in queries:
        try:
            gold[q["id"]] = resolve_gold(q, root, cached_parse)
        except GoldError as e:
            problems.append(str(e))
    if problems:
        print("stale or invalid gold labels:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    print(f"{len(queries)} queries, all gold symbols resolve (repo rev: {repo_rev(root)})")
    if args.check:
        for q in queries:
            spans = ", ".join(f"{g.path.name}:{g.symbol} L{g.start}-{g.end}" for g in gold[q["id"]])
            print(f"  {q['id']}: {spans}")
        return

    # heavy imports only after labels are validated; point maia's indexing at the isolated eval DB
    import context_ret_and_gen as crg
    from index_db import init_db, delete_index
    crg.DB_PATH = EVAL_DB_PATH
    init_db(EVAL_DB_PATH)
    rankers = make_rankers(crg, EVAL_DB_PATH)
    resolver = LineResolver(crg._tokenizer)
    files = collect_py_files(root)
    rev = repo_rev(root)
    print(f"indexing {len(files)} .py files under {root}")

    rows = []
    for cfg_name in args.configs:
        settings = make_settings(cfg_name)
        if args.fresh:
            delete_index(settings["name"], EVAL_DB_PATH)
        t0 = time.time()
        crg._index(files, settings)
        print(f"[{cfg_name}] indexed in {time.time() - t0:.1f}s")

        for ranker_name in args.rankers:
            ranker = rankers[ranker_name]
            for q in queries:
                # score once (the expensive part), then re-rank-evaluate under every de-dup setting
                run_id, scored = ranker({"content": q["query"]}, settings)
                for label in args.dedupe:
                    metrics = evaluate(scored, gold[q["id"]], resolver, args.budgets,
                                       dedupe=parse_threshold(label), cov_threshold=args.cov_threshold)
                    rows.append({"id": q["id"], "config": cfg_name, "ranker": ranker_name,
                                 "dedupe": label, "source": q.get("source", "generated"),
                                 "query_run_id": run_id, "repo_rev": rev, **metrics})

    fields = ["id", "source", "config", "ranker", "dedupe", "first_hit_rank"] + [f"recall@{k}" for k in K_VALUES] + \
             ["mrr", "tokens_to_first_hit"] + [f"budget@{b}" for b in args.budgets] + \
             [f"gold_cov@{b}" for b in args.budgets] + [f"cov80@{b}" for b in args.budgets] + \
             ["top5_mean_lines", "num_chunks", "num_chunks_deduped", "query_run_id", "repo_rev"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in fields})
    print(f"wrote {len(rows)} rows to {args.out}")
    head_rows = [r for r in rows if r["dedupe"] == headline]
    print(f"\n##### full tables at de-dup = {headline} #####")
    print_summary(head_rows)
    print_budget_summary(head_rows, args.budgets)
    print_coverage_summary(head_rows, args.budgets, args.cov_threshold)
    print_bootstrap(head_rows, args.budgets, args.compare, args.boot_samples, args.seed)
    print_source_split(head_rows, args.budgets)
    if len(args.dedupe) > 1:
        print_sweep(rows, args.dedupe, args.budgets)


if __name__ == "__main__":
    main()
