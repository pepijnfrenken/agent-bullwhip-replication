#!/usr/bin/env python3
"""Check interleaved run progress: file size, last records, growth rate."""
import json, glob, os, time

files = glob.glob("results/interleaved_val/*.jsonl")
if not files:
    print("no jsonl")
    raise SystemExit
path = files[0]
size = os.path.getsize(path)
mtime = os.path.getmtime(path)
print(f"file: {os.path.basename(path)} | size: {size} bytes | mtime: {time.ctime(mtime)} | age: {(time.time()-mtime)/60:.1f} min")

lines = [json.loads(l) for l in open(path) if l.strip()]
print(f"records: {len(lines)} | completed: {len([r for r in lines if r.get('total_cost') is not None])} | failed: {len([r for r in lines if r.get('total_cost') is None])}")

print("\nlast 5 records:")
for r in lines[-5:]:
    print(f"  run={r.get('run')} {r.get('config'):<22} cost={r.get('total_cost')} err={str(r.get('error'))[:70]}")
