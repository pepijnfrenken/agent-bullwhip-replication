#!/usr/bin/env python3
"""wave-7 hash forensics — structural hashes over labels.jsonl (stdlib only).

Run (cluster or hub):
    python3 hpc-umbrella/labels_probe.py <path-to-labels.jsonl>

Between two machines running the same labeler code + knobs, these must match:
  J, keys, argflat, ctxparam, tau        (discrete / exact-arithmetic content)
  p_12                                   (p to 12 significant digits; matches
                                          iff float differences are last-ulp)
  row2k                                  (must equal the manifest "hash" value
                                          on the same machine — sanity check)
p_raw / row2k may differ across machines when math.exp differs by 1 ulp.
"""
import hashlib
import json
import sys

path = sys.argv[1]
H = {k: hashlib.sha256() for k in
     ("J", "keys", "argflat", "ctxparam", "tau", "p_12", "p_raw", "row2k")}
n = 0
row0 = None
for line in open(path):
    r = json.loads(line)
    n += 1
    if row0 is None:
        row0 = r
    H["J"].update(json.dumps(r["J"], separators=(",", ":")).encode())
    H["keys"].update(("%d,%d,%s,%s;" % (r["episode"], r["week"], r["role"],
                                        r["curriculum"])).encode())
    H["argflat"].update(("%d,%d,%d;" % (r["argmin_idx"], r["argmin_order"],
                                        1 if r["flat"] else 0)).encode())
    H["ctxparam"].update(json.dumps(r["ctx"], sort_keys=True,
                                    separators=(",", ":")).encode())
    H["ctxparam"].update(json.dumps(r["params"], sort_keys=True,
                                    separators=(",", ":")).encode())
    H["tau"].update(repr(r["tau"]).encode())
    H["p_12"].update(json.dumps(["%.12g" % x for x in r["p"]],
                                separators=(",", ":")).encode())
    H["p_raw"].update(json.dumps(r["p"], separators=(",", ":")).encode())
    if n <= 2000:
        H["row2k"].update(json.dumps(r, sort_keys=True,
                                     separators=(",", ":")).encode())
print("rows    ", n)
if row0 is None:
    sys.exit("FAIL: empty file")
for k in ("J", "keys", "argflat", "ctxparam", "tau", "p_12", "p_raw"):
    print("%-8s" % k, H[k].hexdigest()[:16])
print("row2k   ", H["row2k"].hexdigest())
print("row0 p  ", [float.hex(x) for x in row0["p"][:8]])
print("row0 tau", repr(row0["tau"]))
