"""WAVE8 round orchestrator (docs/waves/WAVE8_SPEC.md §4.1/§4.2/§5, plan unit B2).

One resumable job = one (arm, lr, seed): R rounds of
    collect (argmax + sampled policy rollouts) -> relabel J (26 candidates)
    -> behaviour-mix labels -> merge -> train (exact-gradient RLCD) -> collapse read
and, after the last round, the curriculum-validation eval (seeds 9500..9529) that
selects the lr. Torch-free: every stage is a subprocess (`decision_eval`,
`decision_labeler`, `decision_train`, `scripts/wave8_gates.py`) built by a pure
function, so the hub tests the stage order / resume / SIGTERM / stop-loss with an
injected runner. State lives in `<arm_dir>/round_state.json` (atomic); a resumed job
continues at the first incomplete stage.

Seed bands (disjoint from eval 1000-1029, wave-7 labeler 6000+SEED and collect
7000+SEED, floor tuning 4000-4119, F1 9000-9029, S_nom 9100-9115, validation
9500-9529 — pinned by tests/test_w8_train.py):
    collect(argmax) = 7100 + 10 s + r      collect(sampled) = 7200 + 10 s + r
    label(policy)   = 6100 + 10 s + r      label(behaviour) = 6200 + 10 s + r
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .decision_readout import DEFAULT_GRID_ID, PINNED_REVISION

EX_TEMPFAIL = 75           # SIGTERM-partial (same code decision_train returns)
VAL_SEED_BASE = 9500       # curriculum-validation band 9500..9529
EPISODE_STRIDE = 10_000    # merged label sources are namespaced by episode id


def round_seeds(s: int, r: int) -> dict[str, int]:
    if not (0 <= s <= 9 and 1 <= r <= 9):
        raise ValueError("training seed must be 0..9 and round 1..9")
    return {"collect": 7100 + 10 * s + r, "sampled": 7200 + 10 * s + r,
            "label": 6100 + 10 * s + r, "behavior": 6200 + 10 * s + r}


def lr_tag(lr: float) -> str:
    return "lr" + ("%g" % lr).replace("e-0", "e-").replace("e+0", "e+")


@dataclass
class RoundPlan:
    arm: str                      # "r1" (fresh LoRA, anchor base) | "r2rl" (SFT start, SFT anchor)
    seed: int
    lr: float
    model: str = "Qwen/Qwen3-0.6B"
    revision: str | None = PINNED_REVISION
    rounds: int = 3
    steps_per_round: int = 170
    batch: int = 32
    grid_id: str = "w8_g26"
    n_argmax: int = 15
    n_sampled: int = 15
    n_behavior: int = 10
    permute_p: float = 0.5
    beta_kl: float = 0.02
    m_futures: int = 16
    micro_batch: int = 0
    max_length: int = 1024
    grad_ckpt: bool = True
    save_every: int = 10
    processes: int = 8
    base_adapter: str | None = None            # r2rl: the SFT final checkpoint
    gates_json: str = "results/wave8/gates.json"
    results: str = "results"
    python: str = "python3"
    tag_prefix: str = "w8"

    def __post_init__(self):
        if self.arm not in ("r1", "r2rl"):
            raise ValueError(f"arm must be r1|r2rl, got {self.arm!r}")
        if self.arm == "r2rl" and not self.base_adapter:
            raise ValueError("arm r2rl needs --base-adapter (the SFT final checkpoint)")
        if self.arm == "r1" and self.base_adapter:
            raise ValueError("arm r1 starts from a fresh LoRA: --base-adapter not allowed")

    # -- layout -------------------------------------------------------------- #
    @property
    def name(self) -> str:
        return f"wave8_{self.arm}_{lr_tag(self.lr)}_seed{self.seed}"

    @property
    def arm_dir(self) -> Path:
        return Path(self.results) / self.name

    def round_dir(self, r: int) -> Path:
        """Flat train out dir (one `verify_pull` EXPECT entry per round)."""
        return Path(self.results) / f"{self.name}_round{r}"

    def work(self, r: int) -> Path:
        return self.arm_dir / f"round{r}"

    @property
    def state_path(self) -> Path:
        return self.arm_dir / "round_state.json"

    @property
    def val_tag(self) -> str:
        return f"w8-val-{self.arm}-{lr_tag(self.lr)}-seed{self.seed}"


@dataclass
class Stage:
    key: str                              # "r1:collect_argmax" — unique per round
    argv: list[str] | None = None         # subprocess command ...
    fn: Callable[[], None] | None = None  # ... or an in-process step
    train: bool = False                   # exit 75 = partial (resumable), not failure


# --------------------------------------------------------------------------- #
# file helpers (pure, hub-tested)
# --------------------------------------------------------------------------- #
def concat_jsonl(paths: list[Path], out: Path) -> int:
    n = 0
    with out.open("w") as f:
        for p in paths:
            for line in Path(p).read_text().splitlines():
                if line.strip():
                    f.write(line + "\n")
                    n += 1
    return n


def merge_label_files(paths: list[Path], out: Path) -> dict:
    """Concatenate label jsonl files, rewriting `episode` as episode + 10_000 * k for
    the k-th source so `pi_sft_key` (episode:week:role) stays unique across sources.
    Returns + writes `<out>.manifest.json` {sha256, n, sources}."""
    rows = []
    for k, p in enumerate(paths):
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["episode"] = int(r["episode"]) + EPISODE_STRIDE * k
            rows.append(r)
    keys = [f"{r['episode']}:{r['week']}:{r['role']}" for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("merged label rows are not unique on (episode, week, role) — "
                         "the source files overlap within one namespace")
    raw = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    out.write_text(raw)
    man = {"sha256": hashlib.sha256(raw.encode()).hexdigest(), "n": len(rows),
           "sources": [str(p) for p in paths], "episode_stride": EPISODE_STRIDE}
    out.with_name(out.name + ".manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    return man


def final_checkpoint(round_dir: Path) -> Path:
    """The last adapter dir of a train run (state.json's checkpoint_dir)."""
    st = json.loads((round_dir / "state.json").read_text())
    return round_dir / st["checkpoint_dir"]


def consecutive_trips(reads: list[dict], k: int = 2) -> bool:
    """True iff the last `k` collapse reads all tripped (W8 §6.5 stop-loss)."""
    return len(reads) >= k and all(bool(r.get("collapsed")) for r in reads[-k:])


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, path)


def read_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"completed": [], "round": 1}


# --------------------------------------------------------------------------- #
# stage construction
# --------------------------------------------------------------------------- #
def prev_adapter(plan: RoundPlan, r: int) -> str | None:
    """Adapter whose policy generates round r's states and initialises its training."""
    if r == 1:
        return plan.base_adapter           # None for r1 (untrained), SFT for r2rl
    return f"<final:{r - 1}>"              # resolved when the stage runs (round r-1 is done)


def round_stages(plan: RoundPlan, r: int) -> list[Stage]:
    seeds = round_seeds(plan.seed, r)
    w = plan.work(r)
    py = [plan.python, "-u", "-m"]
    model = ["--model", plan.model] + (["--revision", plan.revision] if plan.revision else [])
    adapter = prev_adapter(plan, r)
    ad = ["--adapter", adapter] if adapter else []

    def collect(mode: str, n: int, seed: int, offset: int, out: str) -> list[str]:
        return (py + ["agent_bullwhip.decision_eval"] + model + ad
                + ["--grid", plan.grid_id, "--collect-states", str(w / out),
                   "--collect-mode", mode, "--n-collect", str(n),
                   "--collect-seed", str(seed), "--collect-offset", str(offset)])

    def label(extra: list[str], out: str, seed: int) -> list[str]:
        return (py + ["agent_bullwhip.decision_labeler", "--grid", plan.grid_id,
                      "--seed", str(seed), "--m-futures", str(plan.m_futures),
                      "--processes", str(plan.processes), "--hash-rows", "2000",
                      "--out", str(w / out)] + extra)

    labels = w / "round_labels.jsonl"
    out_dir = plan.round_dir(r)
    train = (py + ["agent_bullwhip.decision_train"] + model
             + ["--labels", str(labels), "--phase", "rlcd", "--grid", plan.grid_id,
                "--steps", str(plan.steps_per_round), "--batch", str(plan.batch),
                "--lr", ("%g" % plan.lr), "--beta-kl", ("%g" % plan.beta_kl),
                "--permute-p", ("%g" % plan.permute_p),
                "--max-length", str(plan.max_length), "--save-every", str(plan.save_every),
                "--seed", str(plan.seed), "--out", str(out_dir), "--resume",
                "--tag", f"{plan.arm}-{lr_tag(plan.lr)}-seed{plan.seed}-round{r}"])
    if plan.grad_ckpt:
        train.append("--grad-ckpt")
    if plan.micro_batch:
        train += ["--micro-batch", str(plan.micro_batch)]
    stages = [
        Stage(f"r{r}:collect_argmax", collect("argmax", plan.n_argmax, seeds["collect"], 0,
                                              "states_argmax.jsonl")),
        Stage(f"r{r}:collect_sampled", collect("sampled", plan.n_sampled, seeds["sampled"],
                                               plan.n_argmax, "states_sampled.jsonl")),
        Stage(f"r{r}:concat_states",
              fn=lambda: concat_jsonl([w / "states_argmax.jsonl", w / "states_sampled.jsonl"],
                                      w / "states.jsonl")),
        Stage(f"r{r}:label_policy", label(["--states-jsonl", str(w / "states.jsonl")],
                                          "policy_labels", seeds["label"])),
        Stage(f"r{r}:label_behavior", label(["--episodes", str(plan.n_behavior)],
                                            "behavior_labels", seeds["behavior"])),
        Stage(f"r{r}:merge",
              fn=lambda: merge_label_files([w / "policy_labels.jsonl", w / "behavior_labels.jsonl"],
                                           labels)),
    ]
    if plan.arm == "r1":
        train += ["--init", "fresh"] if r == 1 else ["--init", "adapter", "--base-adapter", adapter]
        train += ["--anchor", "base"]
    else:
        # R2+RL: start from the previous adapter; the KL anchor stays the FROZEN SFT adapter
        # evaluated on this round's new states (table computed through ReadoutModel).
        train += ["--init", "adapter", "--base-adapter", adapter,
                  "--anchor", "external", "--anchor-table", str(w / "anchor_table.json")]
        stages.append(Stage(
            f"r{r}:anchor_table",
            py + ["agent_bullwhip.decision_rounds", "anchor-table"] + model
            + ["--adapter", plan.base_adapter, "--labels", str(labels),
               "--grid", plan.grid_id, "--batch", str(plan.batch),
               "--out", str(w / "anchor_table.json")]))
    stages.append(Stage(f"r{r}:train", train, train=True))
    stages.append(Stage(
        f"r{r}:gate",
        [plan.python, "-u", "scripts/wave8_gates.py", "collapse-check"] + model
        + ["--adapter", f"<final:{r}>",
           "--grid", plan.grid_id, "--n", "256", "--frozen", plan.gates_json,
           "--out", str(w / f"collapse_round{r}.json")]))
    return stages


def val_stage(plan: RoundPlan) -> Stage:
    return Stage(
        f"{plan.arm}:validate",
        [plan.python, "-u", "-m", "agent_bullwhip.decision_eval", "--model", plan.model]
        + (["--revision", plan.revision] if plan.revision else [])
        + ["--adapter", f"<final:{plan.rounds}>", "--grid", plan.grid_id,
           "--pattern", "curriculum", "--mode", "argmax", "--runs", "30",
           "--seed-base", str(VAL_SEED_BASE), "--tag", plan.val_tag,
           "--model-name", "qwen3-trained", "--outdir", f"{plan.results}/wave8_val",
           "--traces", "1"])


# --------------------------------------------------------------------------- #
# runner / orchestration
# --------------------------------------------------------------------------- #
class SubprocessRunner:
    """Runs a stage command; forwards SIGTERM to the live child (so decision_train
    checkpoints and exits 75) and remembers that a stop was requested."""

    def __init__(self):
        self.stop = threading.Event()
        self._child: subprocess.Popen | None = None

    def install(self) -> None:
        signal.signal(signal.SIGTERM, self._on_term)

    def _on_term(self, signum, frame):
        self.stop.set()
        if self._child is not None and self._child.poll() is None:
            self._child.send_signal(signal.SIGTERM)

    def __call__(self, argv: list[str]) -> int:
        self._child = subprocess.Popen(argv)
        try:
            return self._child.wait()
        finally:
            self._child = None


def _resolve_deferred(plan: RoundPlan, st: Stage, r: int | None) -> list[str]:
    """Replace `<final:N>` slots by the final checkpoint of round N's train dir."""
    out = []
    for a in st.argv:
        m = re.fullmatch(r"<final:(\d+)>", a)
        out.append(str(final_checkpoint(plan.round_dir(int(m.group(1))))) if m else a)
    return out


def run_plan(plan: RoundPlan, runner: Callable[[list[str]], int],
             stop: threading.Event | None = None) -> int:
    """Run (or resume) the plan. Returns 0 (done or stop-loss), 75 (SIGTERM-partial),
    or the failing child's exit code."""
    stop = stop or getattr(runner, "stop", None) or threading.Event()
    state = read_state(plan.state_path)
    plan.arm_dir.mkdir(parents=True, exist_ok=True)
    reads_path = plan.arm_dir / "collapse_reads.jsonl"

    def done(key: str) -> bool:
        return key in state["completed"]

    def mark(key: str, r: int) -> None:
        state["completed"].append(key)
        state["round"] = r
        write_state(plan.state_path, state)

    for r in range(1, plan.rounds + 1):
        plan.work(r).mkdir(parents=True, exist_ok=True)
        for st in round_stages(plan, r):
            if done(st.key):
                continue
            if stop.is_set():
                return EX_TEMPFAIL
            if st.fn is not None:
                st.fn()
            else:
                rc = runner(_resolve_deferred(plan, st, r))
                if rc == EX_TEMPFAIL or (stop.is_set() and rc != 0):
                    return EX_TEMPFAIL            # state NOT advanced: stage re-runs
                if rc != 0:
                    print(f"FAILED stage {st.key} exit={rc}", flush=True)
                    return rc
            if st.key.endswith(":gate"):
                res = json.loads((plan.work(r) / f"collapse_round{r}.json").read_text())
                res["round"] = r
                with reads_path.open("a") as f:
                    f.write(json.dumps(res, sort_keys=True) + "\n")
            mark(st.key, r)
            if st.key.endswith(":gate") and consecutive_trips(read_jsonl(reads_path)):
                (plan.arm_dir / "STOPPED_collapse.json").write_text(json.dumps(
                    {"stopped_after_round": r, "reason": "collapse gate tripped at two "
                     "consecutive round reads (WAVE8 §6.5)",
                     "reads": read_jsonl(reads_path)}, indent=2, sort_keys=True))
                print(f"STOPPED: collapse gate tripped at two consecutive reads (round {r})",
                      flush=True)
                return 0
    vs = val_stage(plan)
    if not done(vs.key):
        rc = runner(_resolve_deferred(plan, vs, None))
        if rc == EX_TEMPFAIL or (stop.is_set() and rc != 0):
            return EX_TEMPFAIL
        if rc != 0:
            return rc
        mark(vs.key, plan.rounds)
    (plan.arm_dir / "DONE").write_text("ok\n")
    return 0


# --------------------------------------------------------------------------- #
# R2 SFT stage + anchor table (torch lazily)
# --------------------------------------------------------------------------- #
def sft_argv(model: str, revision: str | None, seed: int, lr: float, *, labels: str,
             steps: int = 2827, batch: int = 32, grid_id: str = "w8_g26",
             permute_p: float = 0.5, save_every: int = 250, max_length: int = 1024,
             grad_ckpt: bool = True, micro_batch: int = 0, brier_weight: float = 0.0,
             out: str | None = None, python: str = "python3") -> list[str]:
    out = out or f"results/wave8_r2sft_seed{seed}"
    a = [python, "-u", "-m", "agent_bullwhip.decision_train", "--model", model]
    if revision:
        a += ["--revision", revision]
    a += ["--labels", labels, "--phase", "sft", "--grid", grid_id, "--steps", str(steps),
          "--batch", str(batch), "--lr", "%g" % lr, "--permute-p", "%g" % permute_p,
          "--max-length", str(max_length), "--save-every", str(save_every),
          "--seed", str(seed), "--out", out, "--resume",
          "--tag", f"r2sft-seed{seed}"]
    if grad_ckpt:
        a.append("--grad-ckpt")
    if micro_batch:
        a += ["--micro-batch", str(micro_batch)]
    if brier_weight:
        a += ["--brier-weight", "%g" % brier_weight]
    return a


def compute_anchor_table(model: str, revision: str | None, adapter: str, labels: str,
                         grid_id: str, batch: int, out: str) -> dict:
    """pi_sft over a round's label rows from the SFT adapter (eval mode, no grad),
    persisted with the labels sha — the frozen R2+RL anchor (torch; cluster only)."""
    from .decision_readout import ReadoutModel, render_prompt
    from .decision_train import load_labels, pi_sft_key
    rows, sha = load_labels(labels)
    rm = ReadoutModel(model, adapter_dir=adapter, revision=revision, grid_id=grid_id)
    table: dict[str, list[float]] = {}
    for lo in range(0, len(rows), max(1, batch)):
        chunk = rows[lo:lo + batch]
        prompts = [render_prompt(r["ctx"], rm.tokenizer, grid_id=grid_id) for r in chunk]
        for r, rec in zip(chunk, rm.readout_batch(prompts, require_mass=0.0)):
            k = pi_sft_key(r)
            if k in table:
                raise ValueError(f"duplicate label row {k}")
            table[k] = rec["probs"]
    write_state(Path(out), {"labels_sha": sha, "n": len(table), "adapter": adapter,
                            "anchor": "external", "pi": table})
    return {"n": len(table), "labels_sha": sha}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent_bullwhip.decision_rounds")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run/resume one (arm, lr, seed) round job")
    run.add_argument("--arm", choices=["r1", "r2rl"], required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--lr", type=float, required=True)
    run.add_argument("--model", default="Qwen/Qwen3-0.6B")
    run.add_argument("--revision", default=PINNED_REVISION)
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--steps-per-round", type=int, default=170)
    run.add_argument("--batch", type=int, default=32)
    run.add_argument("--grid", default="w8_g26")
    run.add_argument("--micro-batch", type=int, default=0)
    run.add_argument("--max-length", type=int, default=1024)
    run.add_argument("--grad-ckpt", type=int, choices=[0, 1], default=1)
    run.add_argument("--save-every", type=int, default=10)
    run.add_argument("--processes", type=int, default=8)
    run.add_argument("--base-adapter", default=None)
    run.add_argument("--gates", default="results/wave8/gates.json")
    run.add_argument("--results", default="results")
    run.add_argument("--dry-run", action="store_true",
                     help="print the planned stage commands and exit")
    at = sub.add_parser("anchor-table", help="(torch) frozen-adapter pi table for R2+RL")
    at.add_argument("--model", default="Qwen/Qwen3-0.6B")
    at.add_argument("--revision", default=PINNED_REVISION)
    at.add_argument("--adapter", required=True)
    at.add_argument("--labels", required=True)
    at.add_argument("--grid", default="w8_g26")
    at.add_argument("--batch", type=int, default=32)
    at.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "anchor-table":
        print(compute_anchor_table(a.model, a.revision or None, a.adapter, a.labels, a.grid,
                                   a.batch, a.out))
        return 0
    plan = RoundPlan(arm=a.arm, seed=a.seed, lr=a.lr, model=a.model,
                     revision=a.revision or None, rounds=a.rounds,
                     steps_per_round=a.steps_per_round, batch=a.batch, grid_id=a.grid,
                     micro_batch=a.micro_batch, max_length=a.max_length,
                     grad_ckpt=bool(a.grad_ckpt), save_every=a.save_every,
                     processes=a.processes, base_adapter=a.base_adapter,
                     gates_json=a.gates, results=a.results, python=sys.executable
                     if os.environ.get("W8_USE_SYS_PYTHON") else "python3")
    if a.dry_run:
        for r in range(1, plan.rounds + 1):
            for st in round_stages(plan, r):
                print(st.key, " ".join(st.argv) if st.argv else "<in-process>")
        print(val_stage(plan).key, " ".join(val_stage(plan).argv))
        return 0
    runner = SubprocessRunner()
    runner.install()
    return run_plan(plan, runner, runner.stop)


if __name__ == "__main__":
    raise SystemExit(main())
