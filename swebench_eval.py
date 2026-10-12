#!/usr/bin/env python3
"""
swebench_eval.py — Retrieval-only evaluation of maia's RAG pipeline against SWE-bench Lite.

Does NOT call the LLM. For each (instance, config) pair, scores every chunk in the
checked-out repo against the issue's problem_statement, then checks whether the
top-k retrieved chunks overlap the file(s)/line-range(s) the gold patch actually
touched.

Usage:
    pip install datasets --break-system-packages
    python swebench_eval.py --configs flat-512 token-128-96 ast-py --limit 30

Run from inside your maia project directory (needs context_ret_and_gen.py, index_db.py
on the path).
"""
import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

from datasets import load_dataset

import context_ret_and_gen
from index_db import delete_index, init_db as _init_eval_db

EVAL_DB_PATH = ".swebench_eval.db"
CACHE_DIR = Path(".swebench_cache")

# route THIS script's indexing through an isolated DB — otherwise _index/_score_all_chunks
# write into context_ret_and_gen's own module-level DB_PATH, which is the SAME file
# maia.py's live chat REPL uses (.maia_index.db). Overriding it here before any calls
# keeps eval runs from polluting your real working index.
context_ret_and_gen.DB_PATH = EVAL_DB_PATH
_init_eval_db(EVAL_DB_PATH)
_index = context_ret_and_gen._index
_score_all_chunks = context_ret_and_gen._score_all_chunks
EXCLUDED_DIRS = {".git", ".venv", "node_modules", "__pycache__", "docs", "doc"}

CONFIGS = {
    "flat-512": {"chunker": "fixed", "chunk_size": 512, "overlap": 64,
                 "embedder": "all-MiniLM-L6-v2"},
    "token-128-96": {"chunker": "fixed", "chunk_size": 128, "overlap": 96,
                      "embedder": "all-MiniLM-L6-v2"},
    "ast-py": {"chunker": "ast", "embedder": "all-MiniLM-L6-v2"},
}

K_VALUES = [1, 3, 5, 10]


# ── gold patch parsing ──────────────────────────────────────────────────────

def parse_patch_hunks(patch_text: str) -> dict[str, list[tuple[int, int]]]:
    """
    Parse a unified diff into {file_path: [(start_line, end_line), ...]} of touched
    regions in the PRE-patch (base_commit) file — the '-' side of each hunk.
    """
    result: dict[str, list[tuple[int, int]]] = {}
    current_file = None
    for line in patch_text.splitlines():
        if line.startswith("--- a/"):
            current_file = line[len("--- a/"):]
        elif line.startswith("@@") and current_file:
            m = re.match(r"@@ -(\d+)(?:,(\d+))?", line)
            if m:
                start = int(m.group(1))
                length = int(m.group(2) or 1)
                result.setdefault(current_file, []).append((start, start + max(length, 1)))
    return result


# ── repo checkout, cached via git worktree ──────────────────────────────────

def get_repo_worktree(repo: str, base_commit: str, cache_dir: Path) -> Path:
    # must be absolute: `git -C <bare> ...` changes git's own cwd context before
    # interpreting other paths, so a relative worktree path silently resolves
    # *inside the bare repo's directory* instead of relative to our actual cwd.
    cache_dir = cache_dir.resolve()
    repo_name = repo.split("/")[-1]
    bare = cache_dir / f"{repo_name}.git"
    if not bare.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--bare", f"https://github.com/{repo}.git", str(bare)],
            check=True,
        )
    worktree = cache_dir / f"{repo_name}-{base_commit[:12]}"

    if worktree.exists():
        return worktree

    # clear stale worktree registrations left in the bare repo's own metadata —
    # git remembers a path was registered even after the directory itself is gone
    # (e.g. an interrupted earlier run, or the cache dir was cleared by hand)
    subprocess.run(["git", "-C", str(bare), "worktree", "prune"], check=False)

    try:
        subprocess.run(
            ["git", "-C", str(bare), "worktree", "add", "--detach", str(worktree), base_commit],
            check=True,
        )
    except subprocess.CalledProcessError:
        # last resort: forcibly clear any leftover registration/dir and retry once
        subprocess.run(["git", "-C", str(bare), "worktree", "remove", "--force", str(worktree)],
                        check=False)
        shutil.rmtree(worktree, ignore_errors=True)
        subprocess.run(
            ["git", "-C", str(bare), "worktree", "add", "--detach", str(worktree), base_commit],
            check=True,
        )
    return worktree


def collect_py_files(root: Path) -> list[str]:
    paths = []
    for p in root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in p.parts):
            continue
        paths.append(str(p))
    return paths


# ── metrics ──────────────────────────────────────────────────────────────────

def chunk_overlaps_gold(meta: dict, repo_root: Path, gold: dict[str, list[tuple[int, int]]]) -> tuple[bool, bool]:
    """
    Returns (file_hit, line_hit) for a single chunk's metadata against gold hunks.
    line_hit is only meaningful (and only True) when the chunk carries line info.
    """
    try:
        rel_path = str(Path(meta["path"]).relative_to(repo_root))
    except ValueError:
        rel_path = meta["path"]

    if rel_path not in gold:
        return False, False

    file_hit = True
    line_hit = False
    c_start = meta.get("start_line")
    c_end = meta.get("end_line")
    if c_start is not None and c_end is not None:
        for g_start, g_end in gold[rel_path]:
            if c_start <= g_end and g_start <= c_end:
                line_hit = True
                break
    return file_hit, line_hit


def evaluate_instance(scored_chunks: list[dict], repo_root: Path,
                       gold: dict[str, list[tuple[int, int]]]) -> dict:
    """
    scored_chunks: [{"score": float, "metadata": {...}}, ...] for ALL chunks, unsorted.
    Returns per-k file/line recall (1 or 0) and MRR contributions for this instance.
    """
    ranked = sorted(scored_chunks, key=lambda c: c["score"], reverse=True)

    result = {f"file_recall@{k}": 0 for k in K_VALUES}
    result.update({f"line_recall@{k}": 0 for k in K_VALUES})
    result["file_mrr"] = 0.0
    result["line_mrr"] = 0.0

    first_file_rank = None
    first_line_rank = None
    for rank, c in enumerate(ranked, start=1):
        file_hit, line_hit = chunk_overlaps_gold(c["metadata"], repo_root, gold)
        if file_hit and first_file_rank is None:
            first_file_rank = rank
        if line_hit and first_line_rank is None:
            first_line_rank = rank
        if first_file_rank is not None and first_line_rank is not None:
            break

    if first_file_rank is not None:
        result["file_mrr"] = 1.0 / first_file_rank
        for k in K_VALUES:
            if first_file_rank <= k:
                result[f"file_recall@{k}"] = 1
    if first_line_rank is not None:
        result["line_mrr"] = 1.0 / first_line_rank
        for k in K_VALUES:
            if first_line_rank <= k:
                result[f"line_recall@{k}"] = 1

    return result


# ── main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                         choices=list(CONFIGS.keys()))
    parser.add_argument("--limit", type=int, default=None,
                         help="only run the first N instances (for smoke-testing)")
    parser.add_argument("--out", type=str, default="swebench_eval_results.csv")
    args = parser.parse_args()

    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    if args.limit:
        dataset = dataset.select(range(args.limit))

    rows = []
    for instance in dataset:
        instance_id = instance["instance_id"]
        repo = instance["repo"]
        base_commit = instance["base_commit"]
        query = {"content": instance["problem_statement"]}
        gold = parse_patch_hunks(instance["patch"])
        if not gold:
            continue

        try:
            worktree = get_repo_worktree(repo, base_commit, CACHE_DIR)
        except subprocess.CalledProcessError as e:
            print(f"[skip] {instance_id}: checkout failed ({e})")
            continue

        file_paths = collect_py_files(worktree)
        if file_paths:
            print(f"[debug] {instance_id}: {len(file_paths)} .py files under {worktree}", flush=True)
        if not file_paths:
            all_files = list(worktree.rglob("*"))[:15]
            print(f"[debug] {instance_id}: 0 .py files under {worktree} "
                  f"(dir exists={worktree.exists()}, total entries found={len(all_files)})",
                  flush=True)
            for f in all_files:
                print(f"[debug]   {'DIR ' if f.is_dir() else 'FILE'} {f}", flush=True)
            print(f"[skip] {instance_id}: no .py files found", flush=True)
            continue

        for config_name in args.configs:
            settings = dict(CONFIGS[config_name])
            settings["name"] = f"swebench-{config_name}-{instance_id}"

            scored_chunks = []
            try:
                _index(file_paths, settings)
                obs_noop = lambda *a, **k: None
                _, scored_chunks = _score_all_chunks(query, settings, obs_noop)
                print(f"[debug] {instance_id} / {config_name}: {len(scored_chunks)} scored chunks",
                      flush=True)
            except Exception as e:
                import traceback
                print(f"[skip] {instance_id} / {config_name}: {e}", flush=True)
                traceback.print_exc()
                continue
            finally:
                # keep the eval DB from growing unbounded across 300 instances x N configs
                try:
                    delete_index(settings["name"], EVAL_DB_PATH)
                except Exception:
                    pass

            if not scored_chunks:
                print(f"[skip] {instance_id} / {config_name}: scored_chunks was empty", flush=True)
                continue

            metrics = evaluate_instance(scored_chunks, worktree, gold)
            metrics["instance_id"] = instance_id
            metrics["repo"] = repo
            metrics["config"] = config_name
            metrics["num_chunks"] = len(scored_chunks)
            rows.append(metrics)

            print(f"{instance_id:45s} {config_name:15s} "
                  f"file_recall@5={metrics['file_recall@5']} "
                  f"line_recall@5={metrics['line_recall@5']}")

    if not rows:
        print("No results — check dataset access and repo checkouts.")
        return

    fieldnames = ["instance_id", "repo", "config", "num_chunks"] + \
                 [f"file_recall@{k}" for k in K_VALUES] + \
                 [f"line_recall@{k}" for k in K_VALUES] + \
                 ["file_mrr", "line_mrr"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows to {args.out}")

    # quick aggregate summary per config
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in K_VALUES:
            agg[r["config"]][f"file_recall@{k}"].append(r[f"file_recall@{k}"])
            agg[r["config"]][f"line_recall@{k}"].append(r[f"line_recall@{k}"])
        agg[r["config"]]["file_mrr"].append(r["file_mrr"])
        agg[r["config"]]["line_mrr"].append(r["line_mrr"])

    print("\n=== summary (mean across instances) ===")
    for config_name, metrics in agg.items():
        print(f"\n{config_name}:")
        for metric_name, values in metrics.items():
            print(f"  {metric_name}: {sum(values) / len(values):.3f}  (n={len(values)})")


if __name__ == "__main__":
    main()
