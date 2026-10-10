"""Coding benchmark: base Qwen3-0.6B vs the GRPO-math-finetuned version.

Question: does math GRPO training transfer to code, or is there a
specialization tax?

- Dataset: MBPP sanitized test split (google-research-datasets/mbpp), the
  standard 'basic Python' benchmark — right difficulty for a 0.6B model.
- Same prompt, same decoding (greedy), same test harness for BOTH arms.
  Only difference: LoRA adapter weights from /kaggle/input/grpo-math-lora.
- pass@1: generated code is EXECUTED against the problem's assert tests.
- Function name is given in the prompt for both arms (extracted from the
  reference solution) — otherwise name-guessing luck, not code skill,
  would dominate the score at this model size.
"""
import json
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs],
                   check=False)

_pip("--upgrade", "transformers>=4.51,<5", "peft>=0.15", "datasets>=3.0")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                "torchao", "torchao-nightly"], check=False)

import datasets as ds_lib
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
import glob as _glob
_cands = _glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
LORA_DIR = os.path.dirname(_cands[0]) if _cands else None
print(f"LORA_DIR={LORA_DIR}", flush=True)
MAX_NEW = 256
BATCH = 16
SEED = 0
OUT = "/kaggle/working"
LIMIT = None   # set to an int for a quick subset smoke test

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"cuda={torch.cuda.is_available()}", flush=True)

# ---------------------------------------------------------------- dataset
print("loading MBPP sanitized test...", flush=True)
ds = ds_lib.load_dataset("google-research-datasets/mbpp", "sanitized",
                         split="test")
problems = list(ds)
if LIMIT:
    problems = problems[:LIMIT]
print(f"problems: {len(problems)}", flush=True)


def fname_of(row):
    """Function name the tests will call, taken from the reference code."""
    m = re.match(r"\s*def\s+([A-Za-z_]\w*)", row["code"])
    return m.group(1) if m else None


def build_prompt(task, fname):
    return (
        "# Python function generation\n\n"
        "Example:\n"
        "Task: Write a python function to add two numbers.\n"
        "```python\n"
        "def add_numbers(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        f"Task: {task}\n"
        f"The function must be named `{fname}`.\n"
        f"```python\n"
        f"def {fname}"
    )


def extract_code(gen_text):
    """Extract code from a prefix-completion generation.

    The prompt ends with an UNCLOSED '```python\\ndef {fname}', so the model
    continues the function body and then emits a closing fence. If we naively
    looked for text between the first two fences we would capture the prose
    the model writes AFTER closing (it often hallucinates a whole second
    problem with its own new fence) — that was the 0/215 bug.
    """
    t = gen_text.lstrip()
    if t.startswith("```"):
        # model opened its own fence instead of continuing: content until close
        body = t[3:]
        nl = body.find("\n")
        if nl >= 0 and body[:nl].strip() in ("python", "py", ""):
            body = body[nl + 1:]
        close = body.find("```")
        return body[:close] if close >= 0 else body
    # prefix-completion: code runs until the first closing fence / prose marker
    for stop in ("```", "\nTask:", "\nNow you", "\n# Python"):
        idx = gen_text.find(stop)
        if idx > 0:
            return gen_text[:idx]
    return gen_text


def run_tests(code, test_imports, test_list, timeout=5):
    """Execute generated code + assert tests in a subprocess."""
    program = "\n".join(test_imports or []) + "\n" + code + "\n\n"
    program += "\n".join(test_list)
    try:
        r = subprocess.run([sys.executable, "-c", program],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1][:200]
    except subprocess.TimeoutExpired:
        return False, "timeout"


# ----------------------------------------------------------------- models
def load_base():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32 if device == "cpu" else torch.float16)
    mdl.to(device).eval()
    return tok, mdl


def generate_batch(mdl, tok, prompts):
    enc = tok(prompts, return_tensors="pt", padding=True,
              truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = mdl.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for o in out]


def evaluate_arm(name, mdl, tok):
    rows = []
    prompts, metas = [], []
    for p in problems:
        fname = fname_of(p)
        if not fname:
            continue
        prompts.append(build_prompt(p["prompt"], fname))
        metas.append(p)

    t0 = time.time()
    n_pass = 0
    for i in range(0, len(prompts), BATCH):
        gens = generate_batch(mdl, tok, prompts[i:i + BATCH])
        for gen, p in zip(gens, metas[i:i + BATCH]):
            fname = fname_of(p)
            # prompt ends with 'def {fname}' (unclosed) — the generation is a
            # continuation, so re-attach the header before executing
            code = "def " + fname + extract_code(gen)
            ok, err = run_tests(code, p.get("test_imports"), p["test_list"])
            n_pass += int(ok)
            rows.append({"task_id": p["task_id"], "pass": ok, "err": err,
                         "gen": gen[:600]})
        if (i // BATCH) % 5 == 0:
            print(f"  [{name}] {i+len(gens)}/{len(prompts)} "
                  f"pass_so_far={n_pass}", flush=True)
    acc = n_pass / len(rows)
    print(f"ARM {name}: pass@1 = {n_pass}/{len(rows)} = {acc:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return acc, rows


print("--- ARM 1: BASE (no adapter) ---", flush=True)
tok, base = load_base()
acc_b, rows_b = evaluate_arm("base", base, tok)
# save arm 1 immediately so an arm-2 crash can't lose it
with open(os.path.join(OUT, "arm_base.json"), "w") as f:
    json.dump({"acc": acc_b, "rows": rows_b}, f)
print("SAMPLE generations (base):", flush=True)
for r in rows_b[:3]:
    print(f"  task {r['task_id']} pass={r['pass']} err={r['err']!r}\n"
          f"  gen[:400]={r['gen'][:400]!r}\n", flush=True)
del base
torch.cuda.empty_cache()

print("--- ARM 2: BASE + math GRPO LoRA ---", flush=True)
if LORA_DIR is None:
    print("FATAL: no adapter found under /kaggle/input", flush=True)
    sys.exit(4)
tok, tuned = load_base()
tuned = PeftModel.from_pretrained(tuned, LORA_DIR)
tuned.eval()
acc_t, rows_t = evaluate_arm("base+lora", tuned, tok)

# ------------------------------------------------------- paired analysis
by_b = {r["task_id"]: r["pass"] for r in rows_b}
by_t = {r["task_id"]: r["pass"] for r in rows_t}
both = set(by_b) & set(by_t)
only_base = [t for t in both if by_b[t] and not by_t[t]]
only_tuned = [t for t in both if by_t[t] and not by_b[t]]
neither = len([t for t in both if not by_b[t] and not by_t[t]])
print(f"\npaired: both_pass={len(both)-len(only_base)-len(only_tuned)-neither} "
      f"base_only={len(only_base)} lora_only={len(only_tuned)} "
      f"neither={neither}", flush=True)

results = {"base_acc": acc_b, "lora_acc": acc_t,
           "both_only_base": only_base, "lora_only": only_tuned,
           "base_rows": rows_b, "lora_rows": rows_t}
with open(os.path.join(OUT, "code_bench_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"RESULT base_acc={acc_b:.3f} lora_acc={acc_t:.3f} "
      f"n={len(rows_b)} delta={acc_t-acc_b:+.3f}", flush=True)
