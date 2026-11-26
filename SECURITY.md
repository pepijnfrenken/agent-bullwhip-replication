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

Two scoping facts a reader should know, because both were wrong once:

- The tree scope reads files by suffix. Logs (`.log`, `.out`, `.err`) were missing from that list
  until 2026-11-26, so an earlier "clean" result did not cover `results/wave8_logs/` at all. That is
  where Slurm writes its job header, including the account number. It is now in scope. Current
  result with logs included: 21 identity matches, all cluster hostnames, and no credential-shaped
  string anywhere.
- The account-number patterns are loaded from `TUE_USER` or `~/.config/bullwhip/cluster.env`, never
  written into the scanner. If neither is set the scan prints a warning and does not check for the
  identifier at all, rather than reporting a clean result it did not earn.

API-backed arms in this repo read their keys from the environment at run time
(`FREEINFERENCE_API_KEY` and friends). No key is written into a file, a notebook cell, a log, or a
commit. `.env`, `.env.*` and `.env.local` are git-ignored.

## Account identifiers

The cluster account number this work ran under is replaced with the literal placeholder
`<cluster-account>` throughout the tree. Scripts that need it read `TUE_USER` (or `CLUSTER_USER`)
from the environment and fail loudly when it is unset, rather than shipping a default.

Keeping it out requires an active step, because Slurm stamps the account into every log header and a
retention guard refreshes those logs from the cluster on each cycle. The guard therefore redacts on
copy and refuses to push if any match survives; that check exists because the first version of the
guard wrote the identifier back into 24 files and pushed them here within one cycle of the tunnel
coming back up.

Historical commits are not rewritten, so the identifier remains recoverable from `git log -p`. That
is a deliberate trade: this repository's value is its commit-dated audit trail, and rewriting
history would break every sha the manuscript cites. If that trade is wrong for your use of the
material, say so and it can be re-cut.

## Reporting

If you find a credential, a private identifier, or any other material in this repository that should
not be public, email `pepijnfrenken@mailbox.org` with the path and commit. Do not open an issue for
it. Expect a reply within a few days; anything credential-shaped will be rotated first and discussed
after.
