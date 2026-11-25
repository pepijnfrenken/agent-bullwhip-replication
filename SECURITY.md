# Security and disclosure

This is a research repository: simulation code, run artifacts, and a manuscript. It runs no
service and holds no user data. There is no production system behind it.

## No credentials are stored here

Two things enforce that, and both are meant to be run by a reader:

```bash
python3 scripts/secret_scan.py             # working tree
python3 scripts/secret_scan.py --history    # every blob reachable from any ref
```

`--history` matters because rewriting a file does not rewrite the commits that contain it. The
scanner covers 16 credential formats (private keys, WireGuard keys, GitHub/Hugging Face/OpenAI/
Anthropic/AWS/Google/Slack tokens, JWTs, Modal tokens, W&B keys, generic `token = "..."` assignments)
and prints every match redacted.

API-backed arms in this repo read their keys from the environment at run time
(`FREEINFERENCE_API_KEY` and friends). No key is written into a file, a notebook cell, a log, or a
commit. `.env`, `.env.*` and `.env.local` are git-ignored.

## Account identifiers

The cluster account number this work ran under is replaced with the literal placeholder
`<cluster-account>` throughout the tree. Scripts that need it read `TUE_USER` (or `CLUSTER_USER`)
from the environment and fail loudly when it is unset, rather than shipping a default.

Historical commits are not rewritten, so the identifier remains recoverable from `git log -p`. That
is a deliberate trade: this repository's value is its commit-dated audit trail, and rewriting
history would break every sha the manuscript cites. If that trade is wrong for your use of the
material, say so and it can be re-cut.

## Reporting

If you find a credential, a private identifier, or any other material in this repository that should
not be public, email `pepijnfrenken@mailbox.org` with the path and commit. Do not open an issue for
it. Expect a reply within a few days; anything credential-shaped will be rotated first and discussed
after.
