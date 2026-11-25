#!/usr/bin/env python3
"""secret_scan.py — pre-publication secret and identity scan.

Two scopes, because a working-tree scan is not enough before a repo goes public:
  1. the working tree (tracked and untracked files), and
  2. every blob in the full history, since a rewritten tree does not rewrite history.

Exit 1 on any hit. Matches are printed redacted: first 4 and last 4 characters.

Run:  python3 scripts/secret_scan.py            # tree
      python3 scripts/secret_scan.py --history  # tree + every historical blob
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# High-confidence credential formats first: these should never appear.
CREDENTIALS = {
    "private-key-block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "wireguard-private": r"(?i)\bPrivateKey\s*=\s*[A-Za-z0-9+/]{40,}={0,2}",
    "wireguard-psk": r"(?i)\bPresharedKey\s*=\s*[A-Za-z0-9+/]{40,}={0,2}",
    "github-token": r"\bgh[pousr]_[A-Za-z0-9]{36,}",
    "github-pat": r"\bgithub_pat_[A-Za-z0-9_]{50,}",
    "huggingface-token": r"\bhf_[A-Za-z0-9]{30,}",
    "openai-style-key": r"\bsk-[A-Za-z0-9_-]{20,}",
    "anthropic-key": r"\bsk-ant-[A-Za-z0-9_-]{20,}",
    "aws-key-id": r"\bAKIA[0-9A-Z]{16}\b",
    "aws-secret": r"(?i)aws_secret_access_key\s*[:=]\s*\S{20,}",
    "google-api-key": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "slack-token": r"\bxox[baprs]-[0-9A-Za-z-]{10,}",
    "jwt": r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.",
    "modal-token": r"\bak-[A-Za-z0-9]{20,}\b|\bas-[A-Za-z0-9]{20,}\b",
    "wandb-key": r"(?i)WANDB_API_KEY\s*[:=]\s*\S{20,}",
    "generic-assignment": (
        r"(?i)\b(api[_-]?key|secret|passwd|password|access[_-]?token|auth[_-]?token)"
        r"\s*[:=]\s*[\"'][^\"'\s]{12,}[\"']"
    ),
}

# Identity and infrastructure detail: not credentials, but not for publication.
IDENTITY = {
    "tailnet-ip": r"\b100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.\d{1,3}\.\d{1,3}\b",
    "rfc1918-ip": r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",
    "student-number": r"\b20234428\b",
    "cluster-account-path": r"/home/20234428|~20234428",
    "vpn-config": r"(?i)wireproxy|student-split|tue-wireproxy\.conf",
    "cluster-host": r"\btue-(?:login|cluster|gpua|gpuc|gpub)\d*\b",
}

# Files that legitimately contain such strings in this repo (documented, reviewed).
ALLOW = {
    "scripts/secret_scan.py",  # the patterns themselves
    "docs/paper/paper.tex",  # may quote a filename containing "wireproxy"
}

TEXT_SUFFIXES = {
    ".py", ".sh", ".md", ".json", ".jsonl", ".tex", ".txt", ".yaml", ".yml",
    ".toml", ".cfg", ".conf", ".sbatch", ".bib", ".csv", ".tsv", ".ini", ".env",
}

BINARY_NAME = re.compile(r"(?i)\.(?:safetensors|bin|pt|pth|npz|npy|gguf|tar|tgz|gz|zip|pdf|png|jpg)$")


def redact(s: str) -> str:
    s = s.strip()
    if len(s) <= 8:
        return "*" * len(s)
    return f"{s[:4]}…{s[-4:]}"


def scan_text(label: str, text: str, findings: list[tuple[str, str, int, str, str]]) -> None:
    for table, kind in ((CREDENTIALS, "credential"), (IDENTITY, "identity")):
        for name, pattern in table.items():
            for m in re.finditer(pattern, text):
                line = text.count("\n", 0, m.start()) + 1
                findings.append((kind, name, line, label, redact(m.group(0))))


def scan_tree(root: Path, findings) -> int:
    checked = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".git/") or "/.git/" in rel or rel in ALLOW:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        checked += 1
        scan_text(rel, text, findings)
    return checked


def scan_history(root: Path, findings) -> int:
    """Scan every blob reachable from any ref, including unreachable-but-packed ones."""
    shas = subprocess.run(
        ["git", "-C", str(root), "rev-list", "--objects", "--all"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    paths: dict[str, str] = {}
    for line in shas:
        sha, _, name = line.partition(" ")
        paths[sha] = name or "<unnamed>"

    # add any blob reachable only from the reflog or unreachable objects
    loose = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch-check", "--batch-all-objects"],
        capture_output=True, text=True,
    ).stdout.splitlines()
    for line in loose:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "blob":
            paths.setdefault(parts[0], "<unreachable>")

    blobs = [sha for sha, name in paths.items() if not BINARY_NAME.search(name)]
    proc = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input="\n".join(blobs) + "\n", capture_output=True, text=True,
    )
    out = proc.stdout
    pos = 0
    checked = 0
    while pos < len(out):
        nl = out.find("\n", pos)
        if nl == -1:
            break
        header = out[pos:nl]
        parts = header.split()
        if len(parts) != 3:  # "<sha> missing"
            pos = nl + 1
            continue
        sha, kind, size = parts[0], parts[1], int(parts[2])
        body = out[nl + 1 : nl + 1 + size]
        pos = nl + 1 + size + 1
        if kind != "blob" or "\x00" in body[:4096]:
            continue
        name = paths.get(sha, "<unknown>")
        if name in ALLOW:
            continue
        checked += 1
        scan_text(f"{name} @{sha[:8]}", body, findings)
    return checked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    findings: list[tuple[str, str, int, str, str]] = []
    n_tree = scan_tree(root, findings)
    n_hist = scan_history(root, findings) if args.history else 0

    creds = [f for f in findings if f[0] == "credential"]
    idents = [f for f in findings if f[0] == "identity"]
    print(f"tree files scanned: {n_tree}")
    if args.history:
        print(f"history blobs scanned: {n_hist}")
    print(f"credential hits: {len(creds)}   identity hits: {len(idents)}")

    for kind, name, line, label, sample in creds:
        print(f"CRED {name:22s} {label}:{line}  {sample}")
    if idents:
        by_name: dict[tuple[str, str], int] = {}
        for _, name, line, label, sample in idents:
            by_name[(name, label)] = by_name.get((name, label), 0) + 1
        print("\nidentity hits grouped (count, pattern, file):")
        for (name, label), n in sorted(by_name.items(), key=lambda kv: -kv[1])[:40]:
            print(f"  {n:4d}  {name:18s} {label}")
        print(f"  ({len(by_name)} distinct pattern/file pairs)")

    if creds:
        print("\nFAIL: credential-shaped strings present.")
        return 1
    print("\nclean: no credential-shaped string in scope.")
    return 0 if not idents else 2


if __name__ == "__main__":
    sys.exit(main())
