#!/usr/bin/env python3
"""Check README task status: files produced + process alive."""
import os, glob, time, subprocess

# 1) deliverables
for f in ["README.md", "docs/README_v1.md", "docs/README_v2.md", "docs/README_ITERATION_LOG.md"]:
    if os.path.exists(f):
        print(f"EXISTS {f} ({os.path.getsize(f)} bytes, mtime {time.ctime(os.path.getmtime(f))})")
    else:
        print(f"MISSING {f}")

# 2) any docs dir?
print("docs/:", sorted(glob.glob("docs/*")) if os.path.isdir("docs") else "no docs dir")

# 3) omp proc alive?
r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
omp = [l for l in r.stdout.splitlines() if "omp --model" in l and "grep" not in l]
print("omp procs:", len(omp))
for l in omp[:2]:
    print(" ", l.split()[1], l.split()[10:13])
