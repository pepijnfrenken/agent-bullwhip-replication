#!/usr/bin/env python3
"""wave-7 float-environment fingerprint (stdlib only).

Run:  python3 hpc-umbrella/rng_math_probe.py

The labeler's only transcendental call inside row content is math.exp (softmax
p); math.log/cos/sqrt appear inside random.gauss feeds. This probes each layer
separately, so a difference can be attributed to CPython code (rng layer) vs
the platform libm (math layer).

Hub (py 3.13.12) expected values are in the wave-7 hash-forensics message.

Result (2026-10-06): every fingerprint MATCHED across hub/cluster — rng and libm
are NOT the hash-difference source. It is CPython's builtin sum(): 3.12 switched
float summation to Neumaier (gh-100425); hub 3.13 vs cluster 3.11.3.
See hpc-umbrella/HASH-FORENSICS.md.
"""
import hashlib
import math
import random
import sys


def h(x):
    return hashlib.sha256(repr(x).encode()).hexdigest()[:12]


def line(name, x):
    print("%-8s" % name, h(x))

print(sys.version.split()[0])
r = random.Random(424242); line("random", [r.random() for _ in range(2000)])
r = random.Random(424242); line("uniform", [r.uniform(1.5, 5.0) for _ in range(2000)])
r = random.Random(424242); line("gauss", [r.gauss(0.0, 1.0) for _ in range(20000)])
r = random.Random(424242); line("choice", [r.choice(list(range(0, 41, 2))) for _ in range(2000)])
# exp sweeps cover the softmax's real exponent domain (-(J-Jmin)/tau in ~[-6, 0])
line("exp_dense", [math.exp(-6.0 + i * 0.0001) for i in range(60001)])
rr = random.Random(1234)
line("exp_rand", [math.exp(-6.0 * rr.random()) for _ in range(20000)])
line("log", [math.log(u / 1024.0) for u in range(1, 1025)])
line("cos", [math.cos(math.tau * i / 8000.0) for i in range(8000)])
line("sqrt", [math.sqrt(u / 64.0) for u in range(1, 1025)])
for v in (math.exp(1.0), math.exp(-3.7), math.exp(-17.3), math.log(0.375),
          math.cos(0.5)):
    print("hex", float.hex(v))
