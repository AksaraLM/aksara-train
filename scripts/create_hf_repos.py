#!/usr/bin/env python3
"""Create the private AksaraLLM-20B repos under the Ezekiel999 account.

This is a one-shot helper meant to be run **once** after a fresh HF token is
rotated. Refuses to run if the token is missing or invalid.

    python3 scripts/create_hf_repos.py --dry-run   # print what would happen
    python3 scripts/create_hf_repos.py             # actually create

Reads the token from ``$HF_TOKEN``. Never pass the token on the command line.
"""
from __future__ import annotations

import argparse
import os
import sys

# (repo_id, repo_type)
REPOS: list[tuple[str, str]] = [
    ("Ezekiel999/aksara-pretrain-20b", "dataset"),
    ("Ezekiel999/aksara-sft-20b", "dataset"),
    ("Ezekiel999/aksara-dpo-20b", "dataset"),
    ("Ezekiel999/aksara-tokenizer-20b", "model"),
    ("Ezekiel999/AksaraLLM-20B-Pretrained", "model"),
    ("Ezekiel999/AksaraLLM-20B-Instruct", "model"),
    ("Ezekiel999/AksaraLLM-20B-GGUF", "model"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("error: HF_TOKEN is not set.", file=sys.stderr)
        return 1

    if args.dry_run:
        for repo_id, rt in REPOS:
            print(f"[dry-run] would create_repo({repo_id!r}, type={rt!r}, private=True)")
        print("[dry-run] OK")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        who = api.whoami()
        print(f"Authenticated as {who.get('name')!r}")
    except Exception as e:
        print(f"error: token invalid — {e}", file=sys.stderr)
        return 1

    for repo_id, rt in REPOS:
        try:
            api.create_repo(repo_id=repo_id, repo_type=rt, private=True, exist_ok=True)
            print(f"  ✓ {repo_id} ({rt})")
        except Exception as e:
            print(f"  ✗ {repo_id} ({rt}): {e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
