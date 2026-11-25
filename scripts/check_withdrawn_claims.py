#!/usr/bin/env python3
"""Fail if a withdrawn claim is still asserted anywhere in the documents.

Each row names a claim the manuscript has withdrawn, the exact string that asserts it, and the
files where it must no longer appear. A hit is a build blocker. Historical values are allowed
only where the document explicitly records them as history, and those places are listed in
docs/verify/WITHDRAWN-ALLOW.txt as  <pattern-id>\t<path>\t<reason>.

Usage:  python3 scripts/check_withdrawn_claims.py [--root .]
Exit:   0 clean, 1 hits, 2 allowlist error.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# id: (regex that asserts the withdrawn claim, human label, paths it must not appear in)
WITHDRAWN = {
    "oct4b-3030": (
        r"30/30 per seed|30/30 (?:paired )?wins per seed|"
        r"8\{,\}315[^\n]{0,140}30/30|7\{,\}612[^\n]{0,140}30/30|"
        r"30/30[^\n]{0,140}(?:8\{,\}315|7\{,\}612)|"
        r"30/30[^\n]{0,60}verdict doc|verdict doc[^\n]{0,60}30/30",
        "October 4B paired win count, withdrawn: the verdict document has costs only",
        ["docs/paper/paper.tex", "docs/paper/paper-supplement.tex", "README.md", "docs/STATE.md"],
    ),
    "old-headline": (
        r"1970s|orders of magnitude less compute",
        "the withdrawn wave-3 headline: the rule is Chen et al. (2000) form, and no compute "
        "multiple was ever measured",
        ["README.md", "docs/STATE.md", "docs/paper/paper.tex", "docs/paper/paper-supplement.tex"],
    ),
    "ledger-1113": (
        r"11 with|13 without",
        "8B survivor counted as unevaluated; the ledger is 12/12 after the November read",
        ["docs/paper/paper.tex", "docs/paper/paper-supplement.tex"],
    ),
    "8b-bandonly": (
        r"band-only(?!\.)|no matched control",
        "8B survivor called band-only or without a matched control; it has a 29/30 paired read",
        ["docs/paper/paper-supplement.tex"],
    ),
    "8b-nov-pass": (
        r"both passing G1|two seeds, both pass|selected rate pass(?:es|ed)? G1",
        "November selected-rate 8B legs claimed as a G1 pass; the pairing is invalid",
        ["docs/paper/paper.tex", "docs/paper/paper-supplement.tex", "docs/waves/W8-8B-MOLAB-LEG1.md"],
    ),
    "batches-round": (
        r"162 batches per\s+round|162 batches per round",
        "162 is batches per epoch, not per round",
        ["docs/paper/paper.tex", "docs/paper/paper-supplement.tex"],
    ),
    "oom-seven": (
        r"seven (?:CUDA )?OOM|seven out-of-memory|seven of nine steps",
        "the failed 8B job logged five OOM tracebacks, not seven",
        ["docs/paper/paper.tex", "docs/paper/paper-supplement.tex"],
    ),
    "perm-40781": (
        r"0\.0781|40 of 512|40/512",
        "permutation count from the failed first job; the final artifact is 38/512",
        ["docs/paper/paper.tex", "docs/paper/paper-supplement.tex"],
    ),
    "spec-byteequal": (
        r"byte-equal(?: cluster kit| spec)|spec file was edited once|byte-identical at",
        "the spec blob differs across commits; only the decision-rule text is invariant",
        ["docs/paper/paper.tex", "docs/paper/paper-supplement.tex"],
    ),
    "resume-two": (
        r"two tasks were resumed|two resumed tasks|two resumes",
        "three tasks were resumed past a stop, two of them by one resubmission",
        ["docs/paper/paper.tex", "docs/paper/paper-supplement.tex"],
    ),
    "0.6b-no-eval": (
        r"No wave-8 0\.6B arm\s+has a matched final evaluation|no wave-8 0\.6B arm has a matched",
        "wave-8 R2-SFT arms were evaluated with matched reads (job 418889)",
        ["docs/paper/paper.tex", "docs/paper/paper-supplement.tex"],
    ),
}

ALLOWLIST = Path("docs/verify/WITHDRAWN-ALLOW.txt")
SCAN_SUFFIXES = {".tex", ".md", ".json"}
SKIP_DIRS = {"archive", ".git", "__pycache__"}


def load_allowlist(root: Path) -> dict[tuple[str, str], list[str]]:
    """Rows are <pattern-id>\\t<path>\\t<reason>\\t<snippet>.

    The snippet must appear literally on the offending line. That is deliberate: a row
    authorises ONE reviewed sentence, not every mention of the string in that file, which
    is how a withdrawn claim once survived four audit rounds behind a single stale row.
    """
    f = root / ALLOWLIST
    allowed: dict[tuple[str, str], list[str]] = {}
    if not f.exists():
        return allowed
    for n, line in enumerate(f.read_text().splitlines(), 1):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            raise SystemExit(
                f"allowlist line {n} needs <pattern-id>\\t<path>\\t<reason>\\t<snippet>: {line!r}"
            )
        allowed.setdefault((parts[0], parts[1].strip()), []).append(parts[3].strip())
    return allowed


def main() -> int:
    root = Path(sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else ".")
    root = root.resolve()
    allowed = load_allowlist(root)
    hits: list[tuple[str, str, int, str]] = []

    checked = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        if any(part in SKIP_DIRS for part in path.parts[len(root.parts):]):
            continue
        if rel in {"scripts/check_withdrawn_claims.py"}:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        checked += 1
        for pid, (pattern, label, targets) in WITHDRAWN.items():
            if rel not in targets:
                continue
            for m in re.finditer(pattern, text):
                start = text.rfind("\n", 0, m.start()) + 1
                end = text.find("\n", m.end())
                line_text = text[start : end if end != -1 else len(text)]
                if any(snip in line_text for snip in allowed.get((pid, rel), [])):
                    continue
                line = text.count("\n", 0, m.start()) + 1
                hits.append((pid, rel, line, m.group(0)))

    print(f"scanned {checked} files against {len(WITHDRAWN)} withdrawn claims")
    if not hits:
        print("clean: no withdrawn claim is asserted anywhere it is not allowed")
        return 0
    for pid, rel, line, text in hits:
        print(f"FAIL {pid}  {rel}:{line}  {text!r}  — {WITHDRAWN[pid][1]}")
    print(f"\n{len(hits)} hit(s). Either fix the sentence or add an allowlist row with a reason.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
