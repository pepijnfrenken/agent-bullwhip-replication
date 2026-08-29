#!/usr/bin/env python3
"""Check audit2 status + agent state."""
import os, glob, time

# 1) AUDIT2.md exists?
p = "AUDIT2.md"
if os.path.exists(p):
    print(f"AUDIT2.md: EXISTS ({os.path.getsize(p)} bytes, mtime {time.ctime(os.path.getmtime(p))})")
else:
    print("AUDIT2.md: not yet")

# 2) any audit2 scratch in /tmp?
for f in sorted(glob.glob("/tmp/*audit2*") + glob.glob("/tmp/*bullwhip*audit2*")):
    print("tmp:", f, os.path.getsize(f) if os.path.isfile(f) else "")

# 3) omp proc alive?
import subprocess
r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
omp = [l for l in r.stdout.splitlines() if "omp --model" in l and "grep" not in l]
print("omp procs:", len(omp))
for l in omp[:3]:
    print(" ", l.split()[1], l.split()[10:13])
