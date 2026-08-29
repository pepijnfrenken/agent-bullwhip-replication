#!/usr/bin/env python3
"""Check audit3 status."""
import os, glob, time, subprocess

if os.path.exists("AUDIT3.md"):
    print(f"AUDIT3.md: EXISTS ({os.path.getsize('AUDIT3.md')} bytes, mtime {time.ctime(os.path.getmtime('AUDIT3.md'))})")
else:
    print("AUDIT3.md: not yet")

r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
omp = [l for l in r.stdout.splitlines() if "omp --model" in l and "grep" not in l]
print("omp procs:", len(omp))
for l in omp[:2]:
    print(" ", l.split()[1], l.split()[10:13])
