"""GRPO from scratch: teach a 0.6B LM (closest real model to "0.7B") to do
GSM8K math with rule-based verifiable rewards. Runs on Kaggle GPU.

Local smoke: GRPO_MODE=local python train_grpo_math.py
Mode is also set from env: GRPO_MODE = local | smoke | full  (default smoke)

Design notes:
- No TRL: the trainer is ~150 lines of plain torch (group-normalized
  advantages, clipped importance ratio, LoRA on a frozen base) — fewer
  version-roulette failure modes on a remote box I can't iterate on.
- fp32 everywhere: T4 has no bf16; fp32 dodges fp16 grad-scale weirdness.
- Reward: 1.0 exact match, 0.15 format floor + up to 0.5 continuous partial
  credit for numeric closeness (a 0.6B base model rarely nails binary reward
  early; without variance within a group the GRPO advantage is exactly 0).
  Reported accuracy is ALWAYS exact-match, never the shaped reward.
- Group = 1 problem x NUM_GEN rollouts, advantage = (r - mean) / std per group.
"""
import json
import math
import os
import random
import re
import subprocess
import sys
import time

# must be set before the first CUDA allocation: avoids fragmentation OOM on
# the T4 when the [B, T, vocab] logits tensor size varies between steps
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

# --------------------------------------------------------------- install
def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs],
                   check=False)

_pip("--upgrade", "transformers>=4.51,<5", "peft>=0.15", "datasets>=3.0")
# Kaggle's preinstalled torchao 0.10 trips peft's LoRA dispatcher (needs
# >0.16) -> ImportError at adapter injection. Remove it: peft then falls
# back to standard nn.Linear LoRA, which is what we want here anyway.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                "torchao", "torchao-nightly"], check=False)

import datasets as ds_lib
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------ config
MODE = os.environ.get("GRPO_MODE", "full")  # local | smoke | full

CFG = {
    # steps, prompts per step, rollouts per prompt, max new tokens, eval N
    "local": dict(steps=1, prompts=2, gens=3, max_new=16, eval_n=4, pool=64),
    "smoke": dict(steps=6, prompts=4, gens=8, max_new=256, eval_n=32, pool=256),
    "full":  dict(steps=150, prompts=4, gens=8, max_new=256, eval_n=100,
                  pool=512),
}[MODE]

MODEL_CANDIDATES = [
    os.environ.get("GRPO_MODEL", ""),
    "Qwen/Qwen3-0.6B-Base",       # closest to the requested 0.7B
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-0.5B",
]
MODEL_CANDIDATES = [m for m in MODEL_CANDIDATES if m]

LR = 5e-5
CLIP = 0.2
GRAD_EPOCHS = 2          # passes over the collected group per env step
MINIBATCH = 2            # sequences per gradient accumulation chunk (T4 16G:
                         # logits are [mb, T, 151k-vocab] — keep mb small)
GRAD_CLIP = 1.0
CORRECT_REWARD = 1.0
FORMAT_REWARD = 0.15
PARTIAL_SCALE = 0.5      # + PARTIAL_SCALE * exp(-rel_err) for parseable miss
ADV_EPS = 1e-6
SEED = 0
SAVE_EVERY = 50          # full mode only
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

INSTR = ("Solve the math problem. Reason step by step. On the LAST line "
         "write the final numeric answer prefixed with ####.\n\n")
EXAMPLE_Q = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast "
             "every morning and bakes muffins for her friends every day with "
             "four. She sells the remainder at the farmers' market daily for "
             "$2 per fresh duck egg. How much in dollars does she make every "
             "day at the farmers' market?")
EXAMPLE_A = ("Eggs sold per day: 16 - 3 - 4 = 9. Revenue: 9 * 2 = 18.\n#### 18")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"MODE={MODE} cfg={CFG} torch={torch.__version__} "
      f"transformers={transformers.__version__} "
      f"cuda={torch.cuda.is_available()}", flush=True)
if not torch.cuda.is_available() and MODE != "local":
    print("FATAL: no CUDA but MODE expects GPU", flush=True)
    sys.exit(2)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), flush=True)


def build_prompt(q: str) -> str:
    return (INSTR + "Example:\nProblem: " + EXAMPLE_Q + "\nAnswer: "
            + EXAMPLE_A + "\n\nProblem: " + q + "\nAnswer:")


def norm_num(s: str) -> str:
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    return re.sub(r"[^0-9.\-]", "", s)


def score(pred_text: str, gold: str):
    """-> (reward, correct). gold = text after '####' in GSM8K."""
    m = re.findall(r"####\s*([^\n]+)", pred_text)
    if not m:
        return 0.0, False
    p, g = norm_num(m[-1]), norm_num(gold)
    if not p or not g:
        return FORMAT_REWARD, False
    if p == g:
        return CORRECT_REWARD, True
    try:
        pf, gf = float(p), float(g)
    except ValueError:
        return FORMAT_REWARD, False
    if abs(pf - gf) < 1e-6:
        return CORRECT_REWARD, True
    rel = abs(pf - gf) / (abs(gf) + 1.0)
    return FORMAT_REWARD + PARTIAL_SCALE * math.exp(-rel), False


# ------------------------------------------------------------------- model
tok = mdl = None
MODEL_ID = "?"
for mid in MODEL_CANDIDATES:
    try:
        print(f"loading model: {mid}", flush=True)
        tok = AutoTokenizer.from_pretrained(mid)
        mdl = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        MODEL_ID = mid
        break
    except Exception as e:  # noqa: BLE001 - fall through candidate chain
        print(f"  failed {mid}: {type(e).__name__}: {e}", flush=True)
if mdl is None:
    print("FATAL: no model could be loaded", flush=True)
    sys.exit(3)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"      # decoder-only batched generation
PAD_ID = tok.pad_token_id
EOS_ID = tok.eos_token_id

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
                  task_type="CAUSAL_LM")
mdl = get_peft_model(mdl, lora)
mdl.to(device)
n_train = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in mdl.parameters())
print(f"MODEL={MODEL_ID} trainable={n_train/1e6:.1f}M / "
      f"total={n_all/1e6:.1f}M", flush=True)

opt = torch.optim.AdamW((p for p in mdl.parameters() if p.requires_grad),
                        lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
warmup = max(1, int(CFG["steps"] * 0.05))


def set_lr(step: int) -> float:
    scale = min(1.0, step / warmup)
    for g in opt.param_groups:
        g["lr"] = LR * scale
    return LR * scale


# ------------------------------------------------------------------- data
print("loading GSM8K...", flush=True)
gsm = ds_lib.load_dataset("openai/gsm8k", "main")
train_rows = [r for r in gsm["train"] if r["question"] != EXAMPLE_Q]
random.Random(SEED).shuffle(train_rows)
if CFG["pool"]:
    train_rows = train_rows[:CFG["pool"]]

test_idx = list(range(len(gsm["test"])))
random.Random(SEED).shuffle(test_idx)
test_rows = [gsm["test"][i] for i in test_idx[:CFG["eval_n"]]]
print(f"train pool={len(train_rows)} eval={len(test_rows)}", flush=True)


def gold_of(row) -> str:
    return row["answer"].split("####")[-1].strip()


# -------------------------------------------------------------- generation
def generate(prompt_texts, max_new, sample: bool):
    """prompt_texts: list[str]. -> (seqs [B,T] long, p_len)."""
    enc = tok(prompt_texts, return_tensors="pt", padding=True,
              truncation=True, max_length=512).to(device)
    p_len = enc["input_ids"].shape[1]
    mdl.config.use_cache = True
    kw = dict(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
              max_new_tokens=max_new, pad_token_id=PAD_ID,
              min_new_tokens=4 if sample else 1)
    if sample:
        kw.update(do_sample=True, temperature=1.0, top_p=1.0)
    else:
        kw.update(do_sample=False)
    with torch.no_grad():
        out = mdl.generate(**kw)
    return out, p_len


def completion_mask(seqs, p_len):
    """[B,T] float: 1.0 on generated tokens (prompt region = 0), up to and
    incl. the first EOS. Must be full-sequence shaped: logprobs_of returns
    [B,T] and multiplies by this."""
    comp = seqs[:, p_len:]
    is_eos = comp == EOS_ID
    first = torch.where(is_eos.any(dim=1),
                        is_eos.float().argmax(dim=1),
                        torch.full((comp.shape[0],), comp.shape[1] - 1,
                                   dtype=torch.int64, device=seqs.device))
    col = torch.arange(comp.shape[1], device=seqs.device).unsqueeze(0)
    tail = (col <= first.unsqueeze(1)).float()       # incl. the EOS token
    full = torch.zeros(seqs.shape, dtype=torch.float32, device=seqs.device)
    full[:, p_len:] = tail
    return full


def logprobs_of(seqs, mask_rows):
    """[B,T] log pi(token_t | prefix) aligned to seqs coords (row 0 unused).
    Uses logsumexp over the vocab instead of materializing a full
    log_softmax copy: with vocab=151936 that duplicate was the OOM trigger
    once completions got long enough."""
    attn = torch.ones_like(seqs)
    logits = mdl(input_ids=seqs, attention_mask=attn).logits[:, :-1].float()
    tgt = seqs[:, 1:].unsqueeze(-1)
    got = logits.gather(-1, tgt).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    zero = torch.zeros(seqs.shape[0], 1, dtype=torch.float32,
                       device=seqs.device)
    full = torch.cat([zero, got], dim=1)                        # [B,T]
    del logits, tgt
    return full * mask_rows


# -------------------------------------------------------------- GRPO step
history = []
results = {"mode": MODE, "model": MODEL_ID, "cfg": CFG}


def grpo_step(batch_rows, step: int):
    prompts = [build_prompt(r["question"]) for r in batch_rows]
    golds = [gold_of(r) for r in batch_rows]
    g = CFG["gens"]

    # ---- collect groups: each group has its OWN length (prompt differs),
    # so everything is right-padded to a global max before concatenating.
    seqs_l, mask_l, adv_l = [], [], []
    rewards, corrects = [], []
    for gi, pr in enumerate(prompts):
        gd = golds[gi]                       # one gold per GROUP, not per batch
        seqs, p_len = generate([pr] * g, CFG["max_new"], sample=True)
        texts = [tok.decode(s[p_len:], skip_special_tokens=True) for s in seqs]
        rs, cs = [], []
        for t in texts:
            rw, ok = score(t, gd)
            rs.append(rw)
            cs.append(ok)
        rs = np.asarray(rs, dtype=np.float64)
        adv = (rs - rs.mean()) / (rs.std() + ADV_EPS)
        mask = completion_mask(seqs, p_len)
        seqs_l.append(seqs)
        mask_l.append(mask)
        adv_l.append(torch.as_tensor(adv, dtype=torch.float32,
                                     device=device).unsqueeze(1))
        rewards.extend(rs.tolist())
        corrects.extend(cs)

    t_max = max(s.shape[1] for s in seqs_l)

    def pad_to(x, val, ref):           # right-pad dim1 to t_max
        if x.shape[1] == t_max:
            return x
        p = torch.full((x.shape[0], t_max - x.shape[1]), val,
                       dtype=ref.dtype, device=x.device)
        return torch.cat([x, p], dim=1)

    seqs = torch.cat([pad_to(s, PAD_ID, s) for s in seqs_l])
    mask = torch.cat([pad_to(m, 0.0, m) for m in mask_l])
    adv = torch.cat([pad_to(a * m, 0.0, m) for a, m in zip(adv_l, mask_l)])

    # old-policy logprobs on the PADDED seqs (causal: pad sits after real
    # tokens, so real-position logprobs are identical to the unpadded pass)
    with torch.no_grad():
        old = torch.cat([logprobs_of(seqs[i:i + MINIBATCH],
                                     mask[i:i + MINIBATCH])
                         for i in range(0, seqs.shape[0], MINIBATCH)])

    loss_val = float("nan")
    idx_all = torch.arange(seqs.shape[0])
    cur_lr = set_lr(step)
    for _ in range(GRAD_EPOCHS):
        perm = idx_all[torch.randperm(len(idx_all))]
        for i in range(0, len(perm), MINIBATCH):
            mb = perm[i:i + MINIBATCH]
            mdl.config.use_cache = False
            new = logprobs_of(seqs[mb], mask[mb])
            diff = (new - old[mb]).clamp(-20.0, 20.0)
            ratio = diff.exp()
            surr = torch.min(ratio * adv[mb],
                             ratio.clamp(1 - CLIP, 1 + CLIP) * adv[mb])
            m = mask[mb] > 0
            if m.sum() == 0:
                continue
            loss = -surr[m].mean()
            if not torch.isfinite(loss):
                print("  WARN non-finite loss, skipping mb", flush=True)
                continue
            loss.backward()
            loss_val = float(loss.detach())
        torch.nn.utils.clip_grad_norm_(
            (p for p in mdl.parameters() if p.requires_grad), GRAD_CLIP)
        opt.step()
        opt.zero_grad(set_to_none=True)
    mdl.config.use_cache = True

    rec = dict(step=step, reward_mean=float(np.mean(rewards)),
               correct=float(np.mean(corrects)),
               fmt=float(np.mean([r >= FORMAT_REWARD for r in rewards])),
               loss=loss_val, lr=cur_lr)
    history.append(rec)
    print(f"step={step} reward={rec['reward_mean']:.3f} "
          f"correct={rec['correct']:.2f} fmt={rec['fmt']:.2f} "
          f"loss={loss_val:.4f} lr={cur_lr:.2e}", flush=True)
    return rec


# ------------------------------------------------------------------ eval
def evaluate(rows):
    mdl.eval()
    correct = 0
    samples = []
    bs = 8
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        seqs, p_len = generate([build_prompt(r["question"]) for r in chunk],
                               CFG["max_new"], sample=False)
        for s, r in zip(seqs, chunk):
            text = tok.decode(s[p_len:], skip_special_tokens=True)
            _, ok = score(text, gold_of(r))
            correct += int(ok)
            if len(samples) < 3:
                samples.append({"q": r["question"][:120], "gen": text[-300:],
                                "correct": ok})
    mdl.train()
    acc = correct / len(rows)
    print(f"EVAL n={len(rows)} exact_match_acc={acc:.3f}", flush=True)
    for s in samples:
        print(f"  [{'OK' if s['correct'] else '--'}] {s['q']}\n"
              f"      ...{s['gen'][-160:]!r}", flush=True)
    return acc, samples


def save_all(tag):
    path = os.path.join(OUT, f"lora_{tag}")
    mdl.save_pretrained(path)
    tok.save_pretrained(path)
    results["history"] = history
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT, "curve.csv"), "w") as f:
        f.write("step,reward_mean,correct,fmt,loss,lr\n")
        for h in history:
            f.write(f"{h['step']},{h['reward_mean']},{h['correct']},"
                    f"{h['fmt']},{h['loss']},{h['lr']}\n")
    print(f"saved ({tag})", flush=True)


# ------------------------------------------------------------------- main
t0 = time.time()
print("--- EVAL BEFORE ---", flush=True)
acc_b, _ = evaluate(test_rows)
results["acc_before"] = acc_b
save_all("init")

print("--- TRAIN ---", flush=True)
step = 0
for step in range(1, CFG["steps"] + 1):
    batch = [train_rows[(step * CFG["prompts"] + k) % len(train_rows)]
             for k in range(CFG["prompts"])]
    ts = time.time()
    grpo_step(batch, step)
    print(f"  step_time={time.time() - ts:.1f}s", flush=True)
    if MODE == "full" and step % SAVE_EVERY == 0:
        save_all(f"step{step}")

print("--- EVAL AFTER ---", flush=True)
acc_a, samples_a = evaluate(test_rows)
elapsed = time.time() - t0
results.update(acc_after=acc_a, elapsed_s=round(elapsed), last_step=step,
               samples=samples_a)
save_all("final")

print(f"RESULT mode={MODE} model={MODEL_ID} acc_before={acc_b:.3f} "
      f"acc_after={acc_a:.3f} steps={step} elapsed={elapsed/60:.1f}min",
      flush=True)
