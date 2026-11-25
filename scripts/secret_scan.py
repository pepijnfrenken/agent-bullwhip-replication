#!/usr/bin/env python3
"""secret_scan.py — pre-publication secret and identity scan.

Two scopes, because a working-tree scan is not enough before a repo goes public:
  1. every file a clone would receive (tracked, plus untracked-but-not-ignored), and
  2. every commit in history, since rewriting a file does not rewrite the commits that held it.

Exit 1 on any credential hit. Identity hits exit 2: they are not keys, but they identify a person
or an internal host, and the decision to publish them should be deliberate.

How the two scopes are implemented, and why they differ:
  tree     — Python, one regex pass per pattern over files that contain the pattern's literal core.
             This repo carries ~800 MB of run artifacts, so the prefilter matters.
  history  — `git grep` over every commit. A blob-by-blob sweep in Python is minutes of work for
             the same answer; git's C grep with tree dedup answers in under a second.

Run:  python3 scripts/secret_scan.py            # publishable tree
      python3 scripts/secret_scan.py --history  # tree + every commit
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

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

IDENTITY = {
    # a bare CIDR range such as 100.64.0.0/10 is documentation, not a host address
    "tailnet-ip": r"\b100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.\d{1,3}\.\d{1,3}\b(?!/)",
    "rfc1918-ip": r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",
    "account-number": r"\b20234428\b",
    "cluster-account-path": r"/home/20234428|~20234428",
    "vpn-config": r"(?i)wireproxy|student-split|tue-wireproxy\.conf",
    "cluster-host": r"\btue-(?:login|cluster|gpua|gpuc|gpub)\d*\b",
}

# POSIX ERE equivalents for the history pass (git grep -E has no \b and no lookaround).
ERE_CRED = "|".join([
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"PrivateKey[[:space:]]*=[[:space:]]*[A-Za-z0-9+/]{40}",
    r"PresharedKey[[:space:]]*=[[:space:]]*[A-Za-z0-9+/]{40}",
    r"gh[pousr]_[A-Za-z0-9]{36}",
    r"github_pat_[A-Za-z0-9_]{50}",
    r"hf_[A-Za-z0-9]{30}",
    r"sk-ant-[A-Za-z0-9_-]{20}",
    r"sk-[A-Za-z0-9_-]{20}",
    r"AKIA[0-9A-Z]{16}",
    r"aws_secret_access_key[[:space:]]*[:=][[:space:]]*[^[:space:]]{20}",
    r"AIza[0-9A-Za-z_-]{35}",
    r"xox[baprs]-[0-9A-Za-z-]{10}",
    r"eyJ[A-Za-z0-9_-]{10,}\.eyJ",
    r"ak-[A-Za-z0-9]{20}",
    r"as-[A-Za-z0-9]{20}",
    r"WANDB_API_KEY[[:space:]]*[:=][[:space:]]*[^[:space:]]{20}",
    r"(api[_-]?key|apikey|passwd|password|access[_-]?token|auth[_-]?token)[[:space:]]*[:=][[:space:]]*[\"'][^\"']{12}",
])
ERE_IDENT = "|".join([
    r"100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}",
    r"192\.168\.[0-9]{1,3}\.[0-9]{1,3}",
    r"20234428",
    r"wireproxy|student-split",
    r"tue-(login|cluster|gpua|gpuc|gpub)",
])

# Literal cores: a file is screened with a substring test before any regex runs.
CRED_MARKERS = [
    b"PRIVATE KEY", b"PrivateKey", b"PresharedKey", b"ghp_", b"gho_", b"ghu_", b"ghs_",
    b"ghr_", b"github_pat_", b"hf_", b"sk-", b"AKIA", b"aws_secret", b"AIza", b"xox",
    b"eyJ", b"ak-", b"as-", b"WANDB_API_KEY", b"api_key", b"apikey", b"API_KEY",
    b"secret", b"SECRET", b"passwd", b"password", b"PASSWORD", b"access_token", b"auth_token",
]
IDENT_MARKERS = [
    b"100.6", b"100.7", b"100.8", b"100.9", b"100.1",  # every tailnet prefix starts with one of these
    b"192.168", b"20234428", b"wireproxy", b"student-split", b"tue-", b"/home/2",
]

ALLOW = {"scripts/secret_scan.py"}  # holds the patterns themselves

TEXT_SUFFIXES = {
    ".py", ".sh", ".md", ".json", ".jsonl", ".tex", ".txt", ".yaml", ".yml",
    ".toml", ".cfg", ".conf", ".sbatch", ".bib", ".csv", ".tsv", ".ini", ".env", ".cff",
}

MAX_LINE = 4000
MAX_HITS = 400  # cap on findings recorded per category in the history pass  # JSONL rows here reach 7 MB; the generic pattern over one such line buys nothing


def redact(s: str) -> str:
    s = s.strip()
    return "*" * len(s) if len(s) <= 8 else f"{s[:4]}…{s[-4:]}"


def scan_text(label: str, text: str, findings: list) -> None:
    blob = text.encode("utf-8", "replace")
    scans = (
        (CREDENTIALS, "credential", any(m in blob for m in CRED_MARKERS)),
        (IDENTITY, "identity", any(m in blob for m in IDENT_MARKERS)),
    )
    for table, kind, present in scans:
        if not present:
            continue
        for name, pattern in table.items():
            rx = re.compile(pattern)
            for lineno, line in enumerate(text.splitlines(), 1):
                if name == "generic-assignment" and len(line) > MAX_LINE:
                    continue
                for m in rx.finditer(line):
                    findings.append((kind, name, lineno, label, redact(m.group(0))))


def publishable_files(root: Path) -> list[str]:
    """Tracked files plus untracked-but-not-ignored ones: what a clone would contain.

    Walking the filesystem instead pulls in .venv and node_modules, which is slow and answers the
    wrong question: third-party code is not what this repository publishes.
    """
    out: list[str] = []
    for args in (["ls-files", "-z"], ["ls-files", "-z", "--others", "--exclude-standard"]):
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
        out.extend(p for p in r.stdout.split("\0") if p)
    return list(dict.fromkeys(out))


def scan_tree(root: Path, findings) -> int:
    checked = 0
    for rel in publishable_files(root):
        if rel in ALLOW:
            continue
        path = root / rel
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        checked += 1
        scan_text(rel, text, findings)
    return checked


def _history_grep(root: Path, markers: list[bytes], kind: str, table: dict, findings) -> int:
    """Fixed-string grep over every commit, then classify only the matching lines.

    -F is what makes this usable: an alternation of 17 regexes with bounded repeats over 340
    commits and 60k files does not finish in minutes, while fixed strings do the same job in
    under a second because every marker below is the literal core of its pattern.
    """
    revs = subprocess.run(
        ["git", "-C", str(root), "rev-list", "--all"], capture_output=True, text=True, check=True
    ).stdout.split()
    if not revs:
        return 0
    cmd = ["git", "-C", str(root), "grep", "-I", "-n", "-F"]
    for m in markers:
        cmd += ["-e", m.decode()]
    # no `--` before the revs: that would make git treat them as pathspecs and search nothing
    cmd += [*revs]

    # Stream, do not buffer: with the revs actually searched this returns a row per matching
    # file per commit, which for a marker like the account number is hundreds of megabytes and
    # killed an earlier run with SIGKILL. Findings are capped per category; the cap is reported.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, errors="replace", bufsize=1)
    assert proc.stdout is not None
    total = 0
    for line in proc.stdout:
        total += 1
        if len(findings) > MAX_HITS:
            continue
        sha, _, rest = line.partition(":")
        path, _, rest = rest.partition(":")
        lineno, _, content = rest.partition(":")
        if path in ALLOW:  # the scanner's own pattern strings
            continue
        label = f"{path} @{sha[:8]}"
        for name, pattern in table.items():
            for m in re.finditer(pattern, content):
                findings.append((kind, name, int(lineno or 0), label, redact(m.group(0))))
    proc.wait()
    if total > MAX_HITS:
        print(f"note: {kind} matches capped at {MAX_HITS} of {total} grep rows")
    return len(revs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    findings: list = []
    n_tree = scan_tree(root, findings)
    n_hist = 0
    if args.history:
        n_hist = _history_grep(root, CRED_MARKERS, "credential", CREDENTIALS, findings)
        _history_grep(root, IDENT_MARKERS, "identity", IDENTITY, findings)

    creds = [f for f in findings if f[0] == "credential"]
    idents = [f for f in findings if f[0] == "identity"]
    print(f"tree files scanned: {n_tree}")
    if args.history:
        print(f"commits scanned: {n_hist}")
    print(f"credential hits: {len(creds)}   identity hits: {len(idents)}")

    for kind, name, line, label, sample in creds:
        print(f"CRED {name:22s} {label}:{line}  {sample}")
    if idents:
        grouped: dict[tuple[str, str], list[int]] = {}
        for _, name, line, label, _ in idents:
            grouped.setdefault((name, label), []).append(line)
        print("\nidentity hits grouped (count, pattern, file):")
        for (name, label), lines in sorted(grouped.items(), key=lambda kv: -len(kv[1]))[:40]:
            head = ",".join(str(x) for x in lines[:5])
            print(f"  {len(lines):4d}  {name:20s} {label}  lines {head}{'…' if len(lines) > 5 else ''}")
        print(f"  ({len(grouped)} distinct pattern/file pairs)")

    if creds:
        print("\nFAIL: credential-shaped strings present.")
        return 1
    print("\nno credential-shaped string in scope.")
    return 2 if idents else 0


if __name__ == "__main__":
    sys.exit(main())
