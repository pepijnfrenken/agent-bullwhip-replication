#!/usr/bin/env python3
"""Recompute tau/p from stored J and hash them exactly like labels_probe.py.

Cross-machine hash forensics for the wave-7 labels (2026-10-06): hub and cluster
label sets differ only at the last ulp in tau/p, because CPython 3.12 switched
builtin sum() for floats to Neumaier summation (gh-100425). Run this script
under an interpreter to see that interpreter's "side":

  py3.12+ (e.g. hub 3.13) -> hub digests     tau 74a3f1b8ebd438ab  p_raw e2088c44da7e6a82  p_12 6f8d4ce9e05a1c39
  py<=3.11 (cluster 3.11.3) -> cluster digests tau d7b73091486a672c  p_raw 837660d6998362a5  p_12 349919a04a313d95

Both sides recomputed from the SAME input file reproduce their machine's digests
bit-for-bit; decisions/J/ctx are identical everywhere. Full write-up:
hpc-umbrella/HASH-FORENSICS.md.

Usage: python3 labels_recon.py <labels.jsonl> [max_rows]
"""
import hashlib
import json
import math
import platform
import sys

path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0

HT = hashlib.sha256()
HP = hashlib.sha256()
HP12 = hashlib.sha256()
n_rows = 0
tau_diff = 0
p_diff = 0
first_tau = []
first_p = []
for line in open(path):
    if limit and n_rows >= limit:
        break
    r = json.loads(line)
    n_rows += 1
    J = r["J"]
    n = len(J)
    mean = sum(J) / n
    std = math.sqrt(sum((j - mean) ** 2 for j in J) / n)
    tau = 0.5 * max(std, 1e-3)
    HT.update(repr(tau).encode())
    if tau != r["tau"]:
        tau_diff += 1
        if len(first_tau) < 5:
            first_tau.append([r["episode"], r["week"], r["role"],
                              repr(r["tau"]), repr(tau)])
    if not r["flat"]:
        jmin = min(J)
        e = [math.exp(-(j - jmin) / tau) for j in J]
        z = sum(e)
        p = [x / z for x in e]
    else:
        p = [1.0 / n] * n
    HP.update(json.dumps(p, separators=(",", ":")).encode())
    HP12.update(json.dumps(["%.12g" % x for x in p], separators=(",", ":")).encode())
    if any(a != b for a, b in zip(p, r["p"])):
        p_diff += 1
        if len(first_p) < 5:
            first_p.append([r["episode"], r["week"], r["role"]])

print("python", sys.version.split()[0], "libc", platform.libc_ver())
print("rows", n_rows)
print("tau_recomputed ", HT.hexdigest()[:16], "| rows tau!=stored:", tau_diff)
print("p_raw_recomputed", HP.hexdigest()[:16], "| rows p!=stored:", p_diff)
print("p_12_recomputed ", HP12.hexdigest()[:16])
for x in first_tau:
    print("TAUEX", x)
for x in first_p[:3]:
    print("PEX", x)
