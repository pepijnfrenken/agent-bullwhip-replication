#!/usr/bin/env python3
"""Stock Qwen3-0.6B letter-logit readout on beer-game states (zero-shot baseline).

Stage-1 readout demo for the decision-native lane: no training, no generation.
Renders state + question + options (A..U = orders 0..40 step 2), reads the
next-token logits of the single-token option letters, softmaxes them.
Exactly the AstroHan/RLCD readout shape. Run: ~/bw6-eval/.venvs/jev/bin/python jevstyle_readout.py
"""
import string
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
GRID = list(range(0, 41, 2))          # 21 orders -> labels A..U
LABELS = list(string.ascii_uppercase)

INSTRUCTION = ("You are running one stage of a four-stage beer supply chain (retailer -> "
               "wholesaler -> distributor -> factory). You decide how many units to order from "
               "your supplier this week. Holding cost is $1/unit/week; backlog cost is $2/unit/week; "
               "shipments take 2 weeks. Choose the best order using only the supplied state. "
               "Reply with exactly one option letter, without explanation.")

STATES = {
"retailer_w6": """Current week: 6
Your stage: RETAILER
On-hand inventory: 12
Backlog (unfilled orders owed to your customer): 3
Outstanding orders (ordered, not yet received): 10
Incoming order last week: 14
Incoming order this week: 16
Your recent orders: [12, 14, 13]""",
"factory_w12": """Current week: 12
Your stage: FACTORY
On-hand inventory: 0
Backlog (unfilled orders owed to your customer): 26
Outstanding orders (ordered, not yet received): 0
Incoming order last week: 22
Incoming order this week: not yet known (you decide on last week's information)
Your recent orders: [8, 9, 10]""",
}

QUESTION = "How many units should this stage order from its supplier this week?"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa").eval()
    # label token ids (single tokens for A..U in Qwen3 chat mode)
    label_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS[:len(GRID)]]
    assert all(len(tokenizer.encode(l, add_special_tokens=False)) == 1 for l in LABELS[:len(GRID)])

    for name, state in STATES.items():
        options = "\n".join(f"{LABELS[i]}. order {v} units" for i, v in enumerate(GRID))
        body = f"Material:\n{state}\n\nQuestion:\n{QUESTION}\n\nOptions:\n{options}"
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0, -1].float()
        scores = logits[label_ids]
        probs = scores.softmax(-1)
        full_mass = (scores.logsumexp(-1) - logits.logsumexp(-1)).exp().item()
        top = sorted(zip(GRID, probs.tolist()), key=lambda t: -t[1])[:6]
        best = max(range(len(GRID)), key=lambda i: probs[i].item())
        print(f"\n=== {name} ===  tokens={inputs.input_ids.shape[1]}")
        print("argmax:", f"{LABELS[best]} = order {GRID[best]}", f"(p={probs[best]:.3f})")
        print("top-6:", [(f'{LABELS[GRID.index(v)]}:{v}', round(p, 3)) for v, p in top])
        print("dist mean (units):", round(sum(v * p for v, p in zip(GRID, probs.tolist())), 2),
              "| candidate letter mass:", round(full_mass, 4))

if __name__ == "__main__":
    main()
