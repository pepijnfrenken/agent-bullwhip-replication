# Wave-7 label-hash forensics: hub vs cluster (2026-10-06)

**Question.** Hub and cluster ran the same labeler code (untouched since `ee63271`, 10-05
09:14; both label jobs after it) with the same knobs (seed 6000, 700 episodes, M=16 CRN,
tau_scale 0.5), yet the manifest hashes differ: hub `45d9f40d…`, cluster `59a6b39d…`
(first-2000-rows hash). Job 383232's exit=1 was unrelated glue (unexpanded `${LABEL_OUT}`
in a quoted heredoc; fixed in the rerun).

**Answer (verified, reproduced bit-for-bit).** CPython 3.12 changed builtin `sum()` for
floats from left-to-right accumulation to Neumaier compensated summation (gh-100425). The
hub runs Python 3.13.12, the cluster 3.11.3. The labeler's row content touches builtin
`sum()` in exactly three float-critical places (`decision_labeler.py`):

- `mean = sum(j_vals) / n`
- `std = math.sqrt(sum((j - mean) ** 2 for j in j_vals) / n)`
- `z = sum(e)` in the softmax denominator

Those sums involve inexact values (squares of inexact deviations; exp of softmax logits),
so the two summation algorithms disagree in the last ulp on some rows. Everything else in
a row is machine-stable: the rng chain is CPython `random.Random` (identical across
versions), `J` values are exact binary fractions (costs are dyadic; `tot +=` manual
accumulation and `tot / 16` are exact), and no numpy is in the row path. Consequently:

- decisions (`argmin_idx`, `argmin_order`, `flat`), `J`, `ctx`, `params`, row keys:
  **bit-identical** across machines;
- `tau`: 25,063 / 95,200 rows differ (<=1 ulp);
- `p`: 54,255 / 95,200 rows differ (<=1 ulp);
- `mean_target_entropy_nats`: 2.2130750053898116 (hub) vs 2.213075005389812 (cluster).

**Evidence.** Recompute `tau`/`p` from each row's stored `J` with the labeler formulas and
digest exactly like `labels_probe.py` (`tau · p_raw · p_12`). Reference digests:
hub `74a3f1b8ebd438ab · e2088c44da7e6a82 · 6f8d4ce9e05a1c39`,
cluster `d7b73091486a672c · 837660d6998362a5 · 349919a04a313d95`.

| runtime (same machine, same glibc 2.43) | tau | p_raw | p_12 | rows tau != stored |
|---|---|---|---|---|
| Python 3.13.12 (venv) | `74a3f1b8ebd438ab` | `e2088c44da7e6a82` | `6f8d4ce9e05a1c39` | 0 (self) |
| Python 3.12.13 | `74a3f1b8ebd438ab` | `e2088c44da7e6a82` | `6f8d4ce9e05a1c39` | 0 |
| Python 3.11.15 | `d7b73091486a672c` | `837660d6998362a5` | `349919a04a313d95` | 25,063 |

Python 3.11.15 on the hub (same glibc as 3.13) reproduces the cluster's three digests
bit-for-bit; 3.12+ reproduces the hub's. Same-machine, same-glibc isolation shows the
glibc version is irrelevant. (An earlier rockylinux:9 container check also matched the
cluster digests; it ran Python 3.9, pre-3.12. Correct result, different reason.)

**Why the micro-probes all passed.** `rng_math_probe.py` fingerprints rng
(`random.random`/`uniform`/`gauss`/`choice`) and libm (dense `exp` sweeps over the real
softmax arg domain, `log`, `cos`, `sqrt`): every value matched across machines, because
none of those code paths is affected by the `sum()` change. The difference lives in a
pure-Python builtin, not in a library function. numpy is not involved (the npz sidecar is
the only numpy use).

**Prompts are unaffected.** `decision_readout` renders only ints (orders, weeks,
observation counters), so the rendered-prompt hash fixture (`b96fb847…`) is
version-stable across machines.

**Repro.**

```bash
# any labels.jsonl with the same J content (e.g. the hub replay copy)
P=/home/pino/.hermes/cache/scratch/wave7_labels_hubcheck.jsonl
.venv/bin/python hpc-umbrella/labels_recon.py "$P"   # -> hub-side digests
python3.11          hpc-umbrella/labels_recon.py "$P"   # -> cluster-side digests
```

`labels_recon.py` prints the three digests, the count of rows where the recomputed `tau`
differs from the stored one, and the first differing rows.

**Implications.**

1. Cross-machine byte-identity of `labels.jsonl` requires the **same Python minor on both
   sides** (>=3.12 with >=3.12, or <=3.11 with <=3.11). Alternative for a future label
   generation only: switch those three sums to `math.fsum` (or manual accumulation),
   which pins the semantics across versions. Do not do that on an already-trained label
   set: it changes stored values.
2. Cross-machine verification of a label set = `tools/label_fingerprint.py` +
   `hpc-umbrella/labels_probe.py`: decisions + `J` identical, `p` equal to 12 significant
   digits. The local full-file `sha256` (`W7_EXPECT_LABEL_SHA256`) remains the trainer
   gate; it is a same-machine integrity gate, not a cross-machine determinism proof.
3. The wave-7 trained artifacts are unaffected: training ran on the cluster's own file
   (`labels_sha d35e8d9b…`) and never depended on the hub copy.
4. The manifest's `mean_target_entropy_nats` carries the same <=1 ulp cross-version noise
   (it sums `-p·log p` over rows).
