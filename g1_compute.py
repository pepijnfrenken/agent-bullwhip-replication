import json, statistics

def load(f):
    return [json.loads(l) for l in open(f)]

def costs(rows):
    c=[]
    for r in rows:
        v = r.get("cost") or r.get("total_cost") or (r.get("summary",{}) or {}).get("cost")
        if v is None: continue
        c.append(float(v))
    return c

u = costs(load("w8-untrained-4b-argmax-noisy.jsonl"))
s0 = costs(load("w8-r1-4b-seed0-argmax-noisy.jsonl"))
s1 = costs(load("w8-r1-4b-seed1-argmax-noisy.jsonl"))
print("n untrained", len(u), "seed0", len(s0), "seed1", len(s1))

def paired(a, b, label):
    n = min(len(a), len(b))
    red = 1 - statistics.median(b[:n]) / statistics.median(a[:n])
    wins = sum(1 for ca, cb in zip(a[:n], b[:n]) if cb < ca)
    per = statistics.median([(ca - cb) / ca for ca, cb in zip(a[:n], b[:n]) if ca])
    print(f"{label}: med_untr {statistics.median(a[:n]):.0f} med_tr {statistics.median(b[:n]):.0f} ratio_red {red:.3f} perpath_red {per:.3f} wins {wins}/{n}")

paired(u, s0, "seed0 vs untr noisy")
paired(u, s1, "seed1 vs untr noisy")
