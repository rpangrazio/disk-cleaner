#!/usr/bin/env python3
"""
disk_cleaner.py

CLI tool that analyzes a folder and finds duplicate or superseded files,
suggests reorganizations, and can copy/move files into a new structure.

It supports an "OpenAI-style" API for embedding-based clustering. You can
specify a base address, token and model. Requests are batched to save traffic.

Usage examples:
  python disk_cleaner.py --source /path/to/folder --report report.json
  python disk_cleaner.py --source . --apply --mode copy --dest ../cleaned \
      --api-base https://api.openai.com --api-token $TOKEN --model text-embedding-3-small

Defaults to dry-run; use --apply to perform filesystem changes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import requests
except Exception:
    requests = None


CHUNK_SIZE = 8192


def compute_md5(path: str, chunk_size: int = CHUNK_SIZE) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def is_text_file(path: str) -> bool:
    # Quick heuristic based on mimetype
    mt, _ = mimetypes.guess_type(path)
    if mt is None:
        return False
    return mt.startswith("text") or mt in ("application/json", "application/javascript")


def read_text(path: str, max_bytes: int = 1024 * 1024) -> str:
    # Read up to max_bytes to avoid huge payloads
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)
    except Exception:
        return ""


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\x00-\x7f]", "", s)
    return s.strip()


def text_similarity(a: str, b: str) -> float:
    # lightweight similarity using sequence matcher ratio
    try:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0


def cosine_similarity(a: List[float], b: List[float]) -> float:
    da = sum(x * x for x in a)
    db = sum(x * x for x in b)
    if da == 0 or db == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(da) * math.sqrt(db))


class OpenAIStyleClient:
    def __init__(self, base: str, token: str, model: str, batch_size: int = 16):
        self.base = base.rstrip("/")
        self.token = token
        self.model = model
        self.batch_size = batch_size
        if requests is None:
            raise RuntimeError("requests library is required for API calls")

    def _request(self, path: str, body: dict) -> dict:
        url = f"{self.base.rstrip('/')}{path}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        r = requests.post(url, json=body, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def embeddings(self, inputs: List[str]) -> List[List[float]]:
        # Batch inputs to reduce round trips
        out = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i : i + self.batch_size]
            body = {"model": self.model, "input": batch}
            # Support both OpenAI and compatible endpoints (/v1/embeddings)
            resp = self._request("/v1/embeddings", body)
            if "data" not in resp:
                raise RuntimeError("Invalid embedding response: missing data")
            for item in resp["data"]:
                vec = item.get("embedding") or item.get("vector")
                out.append(vec)
        return out


def scan_files(root: str, follow_symlinks: bool = False) -> List[Dict]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
            except Exception:
                continue
            files.append(
                {
                    "path": full,
                    "name": fn,
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                }
            )
    return files


def find_duplicates_by_hash(files: List[Dict], md5_for_all: bool = False, size_threshold: int = 1024 * 8) -> Dict[str, List[str]]:
    # compute md5 for files larger than size_threshold or when requested
    groups = defaultdict(list)
    for f in files:
        use_md5 = md5_for_all or f["size"] >= size_threshold
        if use_md5:
            try:
                h = compute_md5(f["path"])
            except Exception:
                h = None
            if h:
                groups[h].append(f["path"])
    # return only groups with duplicates
    return {k: v for k, v in groups.items() if len(v) > 1}


def find_similar_by_name(files: List[Dict]) -> Dict[str, List[str]]:
    # strip common version tokens and group by base name
    def strip_version(name: str) -> str:
        name_noext, _ = os.path.splitext(name)
        # patterns like _v2, -v2, v2, v1.0, _final, -final, copy, (1)
        name2 = re.sub(r"[._\- ]?(v\d+(?:\.\d+)*)", "", name_noext, flags=re.I)
        name2 = re.sub(r"[._\- ]?(final|copy|backup|bak|old)\b", "", name2, flags=re.I)
        name2 = re.sub(r"\(\d+\)", "", name2)
        return name2.lower().strip()

    groups = defaultdict(list)
    for f in files:
        base = strip_version(f["name"])
        groups[base].append(f["path"])
    return {k: v for k, v in groups.items() if len(v) > 1}


def find_similar_by_content(files: List[Dict], small_size_limit: int = 1024 * 256, threshold: float = 0.85) -> List[Tuple[str, str, float]]:
    # Compare small text files using normalized text similarity
    text_files = [f for f in files if f["size"] <= small_size_limit and is_text_file(f["path"]) ]
    results = []
    texts = {}
    for f in text_files:
        t = normalize_text(read_text(f["path"]))
        texts[f["path"]] = t
    paths = list(texts.keys())
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            a = texts[paths[i]]
            b = texts[paths[j]]
            if not a or not b:
                continue
            sim = text_similarity(a, b)
            if sim >= threshold:
                results.append((paths[i], paths[j], sim))
    return results


def cluster_by_embeddings(client: OpenAIStyleClient, files: List[Dict], max_file_bytes: int = 1024 * 512, similarity_threshold: float = 0.80) -> Dict[int, List[str]]:
    # Prepare texts (only reasonably sized text files)
    items = []
    paths = []
    for f in files:
        if f["size"] > max_file_bytes:
            continue
        if not is_text_file(f["path"]):
            continue
        txt = normalize_text(read_text(f["path"], max_bytes=max_file_bytes))
        if txt:
            items.append(txt)
            paths.append(f["path"])
    if not items:
        return {}

    vecs = client.embeddings(items)

    clusters: Dict[int, List[str]] = {}
    centroids: Dict[int, List[float]] = {}
    next_cluster = 0
    for path, vec in zip(paths, vecs):
        assigned = False
        for cid, centroid in centroids.items():
            sim = cosine_similarity(vec, centroid)
            if sim >= similarity_threshold:
                clusters[cid].append(path)
                # update centroid (incremental mean)
                n = len(clusters[cid])
                centroids[cid] = [(c * (n - 1) + v) / n for c, v in zip(centroid, vec)]
                assigned = True
                break
        if not assigned:
            clusters[next_cluster] = [path]
            centroids[next_cluster] = vec[:]
            next_cluster += 1
    # prune singletons if desired; keep all for now
    return clusters


def propose_reorganization(files: List[Dict], clusters: Optional[Dict[int, List[str]]] = None) -> Dict[str, List[str]]:
    proposals = {}
    # by type (extension)
    by_ext = defaultdict(list)
    for f in files:
        ext = os.path.splitext(f["name"])[1].lower().lstrip('.') or "no_ext"
        by_ext[ext].append(f["path"])
    proposals["by_extension"] = {k: v for k, v in by_ext.items()}

    # by year-month
    by_date = defaultdict(list)
    for f in files:
        dt = datetime.fromtimestamp(f["mtime"]).strftime("%Y-%m")
        by_date[dt].append(f["path"])
    proposals["by_date"] = {k: v for k, v in by_date.items()}

    # include provided clusters
    if clusters:
        proposals["by_cluster"] = clusters

    # by size ranges
    by_size = defaultdict(list)
    for f in files:
        s = f["size"]
        if s < 1024:
            k = "tiny"
        elif s < 1024 * 100:
            k = "small"
        elif s < 1024 * 1024:
            k = "medium"
        else:
            k = "large"
        by_size[k].append(f["path"])
    proposals["by_size"] = {k: v for k, v in by_size.items()}

    return proposals


def build_plan_for_proposals(proposals: Dict[str, List], dest_root: str, mode: str = "copy") -> List[Dict]:
    # Build a simple plan: choose 'by_extension' as default structure
    plan = []
    ext_map = proposals.get("by_extension", {})
    for ext, paths in ext_map.items():
        target_dir = os.path.join(dest_root, ext)
        for p in paths:
            target = os.path.join(target_dir, os.path.basename(p))
            plan.append({"src": p, "dst": target, "action": mode})
    return plan


def apply_plan(plan: List[Dict], dry_run: bool = True) -> List[Dict]:
    results = []
    for step in plan:
        src = step["src"]
        dst = step["dst"]
        action = step.get("action", "copy")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ok = False
        errmsg = None
        try:
            if not dry_run:
                if action == "copy":
                    shutil.copy2(src, dst)
                elif action == "move":
                    shutil.move(src, dst)
                elif action == "delete":
                    os.remove(src)
                    ok = True
                else:
                    errmsg = f"unknown action {action}"
            ok = True
        except Exception as e:
            errmsg = str(e)
        results.append({"src": src, "dst": dst, "action": action, "ok": ok, "error": errmsg})
    return results


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Analyze and reorganize files; detect duplicates and suggest structure")
    p.add_argument("--source", required=True, help="Source directory to analyze")
    p.add_argument("--dest", help="Destination root for reorganized files (used with --apply and --mode copy|move)")
    p.add_argument("--mode", choices=["copy", "move", "delete"], default="copy", help="Operation when applying plan")
    p.add_argument("--apply", action="store_true", help="Apply the plan (default is dry-run)")
    p.add_argument("--report", help="Path to write JSON report of findings and plan")
    p.add_argument("--api-base", help="OpenAI-style API base URL (e.g. https://api.openai.com)")
    p.add_argument("--api-token", help="API token for OpenAI-style API")
    p.add_argument("--model", default="text-embedding-3-small", help="Embedding model name")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for API requests")
    p.add_argument("--similarity-threshold", type=float, default=0.85, help="Thresholds for text similarity")
    p.add_argument("--cluster-similarity", type=float, default=0.80, help="Cosine threshold for embedding clustering")
    p.add_argument("--md5-threshold", type=int, default=8192, help="File size (bytes) above which md5 hashing is used for duplicates")
    p.add_argument("--md5-all", action="store_true", help="Compute MD5 for all files (may be slow) and detect exact duplicates")
    p.add_argument("--yes", action="store_true", help="Agree to perform actions when --apply is set")
    p.add_argument("--dry-run", action="store_true", help="Do not perform any file operations; show the full plan and size estimates")
    args = p.parse_args(argv)

    src = args.source
    if not os.path.isdir(src):
        print(f"source directory not found: {src}")
        return 2

    files = scan_files(src)

    total_size = sum(f["size"] for f in files)


    dup_hash = find_duplicates_by_hash(files, md5_for_all=args.md5_all, size_threshold=args.md5_threshold)
    dup_name = find_similar_by_name(files)
    sim_content = find_similar_by_content(files, small_size_limit=1024 * 256, threshold=args.similarity_threshold)

    clusters = None
    if args.api_base and args.api_token:
        client = OpenAIStyleClient(args.api_base, args.api_token, args.model, batch_size=args.batch_size)
        try:
            clusters = cluster_by_embeddings(client, files, max_file_bytes=1024 * 512, similarity_threshold=args.cluster_similarity)
        except Exception:
            clusters = None

    proposals = propose_reorganization(files, clusters=clusters)

    plan = []
    if args.dest:
        plan = build_plan_for_proposals(proposals, args.dest, mode=args.mode)

    # Build a privacy-preserving report: redact file paths in duplicate groups
    def redact_paths(paths: List[str]) -> List[str]:
        return [os.path.basename(p) for p in paths]

    duplicates_by_hash_redacted = {}
    for h, paths in dup_hash.items():
        duplicates_by_hash_redacted[h] = {"count": len(paths), "files": redact_paths(paths)}

    duplicates_by_name_redacted = {}
    for name, paths in dup_name.items():
        duplicates_by_name_redacted[name] = {"count": len(paths), "files": redact_paths(paths)}

    # Build proposals summary (counts per bucket) to avoid listing paths
    proposals_summary = {}
    for key, buckets in proposals.items():
        if isinstance(buckets, dict):
            proposals_summary[key] = {b: len(v) for b, v in buckets.items()}
        else:
            # if clusters or unexpected format, provide counts
            try:
                proposals_summary[key] = {"count": len(buckets)}
            except Exception:
                proposals_summary[key] = {}

    report = {
        "summary": {"total_files": len(files), "duplicate_hash_groups": len(dup_hash), "duplicate_name_groups": len(dup_name), "similar_content_pairs": len(sim_content)},
        "sizes": {},
        "duplicates_by_hash": duplicates_by_hash_redacted,
        "duplicates_by_name": duplicates_by_name_redacted,
        "similar_content_pairs": sim_content,
        "proposals_summary": proposals_summary,
        # For privacy we do not include the full per-file plan; only a summary below
    }

    # Size calculations: total size, expected after exact dedupe, potential name-based savings
    def human(n: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024.0:
                return f"{n:3.1f}{unit}"
            n /= 1024.0
        return f"{n:.1f}PB"

    total_saved_by_hash = 0
    # For each duplicate hash group, all files are same size; pick one to keep
    for h, paths in dup_hash.items():
        try:
            sz = os.path.getsize(paths[0])
        except Exception:
            sz = 0
        total_saved_by_hash += (len(paths) - 1) * sz

    # potential savings from name-based groups (excluding files already in hash duplicates)
    files_in_hash = set(p for paths in dup_hash.values() for p in paths)
    potential_name_savings = 0
    for group_paths in dup_name.values():
        # exclude paths already accounted
        gp = [p for p in group_paths if p not in files_in_hash]
        if len(gp) <= 1:
            continue
        sizes = []
        for p in gp:
            try:
                sizes.append(os.path.getsize(p))
            except Exception:
                sizes.append(0)
        potential_name_savings += sum(sizes) - max(sizes)

    expected_after_hash = total_size - total_saved_by_hash
    expected_after_name = expected_after_hash - potential_name_savings

    # Plan sizes: current plan copies all files; compute unique size if dedup applied
    # Unique by hash: for each hash group keep one file; for files not in any hash group, include their size
    unique_size = 0
    handled = set()
    for h, paths in dup_hash.items():
        try:
            unique_size += os.path.getsize(paths[0])
        except Exception:
            pass
        handled.update(paths)
    for f in files:
        if f["path"] in handled:
            continue
        unique_size += f["size"]

    report["sizes"] = {
        "total_bytes": total_size,
        "total_human": human(total_size),
        "expected_after_exact_dedup_bytes": expected_after_hash,
        "expected_after_exact_dedup_human": human(expected_after_hash),
        "potential_name_based_savings_bytes": potential_name_savings,
        "potential_name_based_savings_human": human(potential_name_savings),
        "expected_after_name_dedupe_bytes": expected_after_name,
        "expected_after_name_dedupe_human": human(expected_after_name),
        "plan_bytes_if_copy_all": total_size,
        "plan_bytes_unique_if_deduped": unique_size,
        "plan_bytes_unique_if_deduped_human": human(unique_size),
    }

    # Plan summary (no per-file paths)
    plan_summary = {}
    if plan:
        action_counts = defaultdict(int)
        plan_total_bytes = 0
        for step in plan:
            action_counts[step.get("action", "copy")] += 1
            try:
                plan_total_bytes += os.path.getsize(step["src"])
            except Exception:
                pass
        plan_summary = {
            "items": len(plan),
            "actions": dict(action_counts),
            "total_bytes": plan_total_bytes,
            "total_human": human(plan_total_bytes),
        }
    report["plan_summary"] = plan_summary

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    # Compute files affected (union of files flagged by any detection) and space impact
    files_flagged = set()
    for paths in dup_hash.values():
        files_flagged.update(paths)
    for paths in dup_name.values():
        files_flagged.update(paths)
    for a, b, _ in sim_content:
        files_flagged.add(a)
        files_flagged.add(b)

    space_affected_bytes = total_saved_by_hash + potential_name_savings

    # Only print a concise summary to stdout: number of files affected and space affected
    try:
        print(f"Files affected: {len(files_flagged)}")
        print(f"Potential space affected: {human(space_affected_bytes)} ({space_affected_bytes} bytes)")
    except Exception:
        # Avoid raising in case stdout isn't available
        pass

    if args.apply or args.dry_run:
        # determine whether we're actually performing filesystem operations
        perform_ops = args.apply and not args.dry_run
        if perform_ops and not args.dest and args.mode in ("copy", "move"):
            print("--apply with mode copy/move requires --dest")
            return 3
        if perform_ops and not args.yes:
            ans = input("Proceed to apply the plan? [y/N] ")
            if ans.lower() != "y":
                print("Aborted by user")
                return 0
        if args.dry_run and not perform_ops:
            # For privacy, never print the full per-file plan; only show summary/estimates
            pass
        else:
            print("Applying plan...")
        results = apply_plan(plan, dry_run=not perform_ops)
        success = sum(1 for r in results if r.get("ok"))
        # do not print per-file or step counts; keep silent here

    return 0


if __name__ == "__main__":
    sys.exit(main())
