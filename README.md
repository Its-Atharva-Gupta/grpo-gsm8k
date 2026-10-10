# GRPO from scratch — teaching a 0.6B model math on one free T4

Complete, reproducible code + results for **[Atharva1232/Qwen3-0.6B-GRPO-GSM8K](https://huggingface.co/Atharva1232/Qwen3-0.6B-GRPO-GSM8K)** — a Qwen3-0.6B base model taken from **8% → 59%** on GSM8K using GRPO written from scratch in ~150 lines of PyTorch. No TRL, no reward model, no critic, no multi-GPU. **3.2 hours on a single free Kaggle T4.**

```
GSM8K exact-match (100 held-out problems, greedy)
  base model   ██░░░░░░░░░░░░░░░░░░   8%
  + GRPO       ████████████░░░░░░░░  59%    ← 150 steps, 10.1M LoRA params
```

## What's here

| path | what it is |
|---|---|
| [`training/train_grpo_math.py`](training/train_grpo_math.py) | The whole GRPO trainer — data, reward, policy gradient, eval. Runs on Kaggle GPU or CPU for dry runs. |
| [`benchmark/bench_code.py`](benchmark/bench_code.py) | Transfer experiment: does math training help or hurt code ability? Paired MBPP benchmark with code execution. |
| [`results/`](results/) | Raw outputs: training history, learning curve, per-problem benchmark results. |
| [`docs/model_card.md`](docs/model_card.md) | The HuggingFace model card (also rendered on the model page). |
| `*/kernel-metadata.json` | Kaggle kernel configs — push either script to Kaggle with `kaggle kernels push`. |

## How it works

GRPO (Group Relative Policy Optimization) is PPO without the critic:

1. Sample a **group of 8 rollouts** for one problem
2. Score each rollout with a **rule-based verifier** (extract the number after `####`, compare to gold)
3. Normalize within the group: `advantage = (reward - mean) / std`
4. Clipped importance-ratio update over every token/step in the group

No value network, no GAE, no learned reward model — the reward is objective because GSM8K answers are single numbers.

### The part that actually mattered: reward shaping

A cold 0.6B base model almost never emits a parseable answer. And if all 8 rollouts in a group get the *same* reward, GRPO's advantage is exactly **zero** — no gradient, no learning. So the reward is shaped:

| condition | reward |
|---|---|
| exact match | `1.0` |
| parseable `#### <number>`, wrong value | `0.15` (format floor) |
| numeric miss | `+ 0.5 · exp(−relative_error)` (partial credit) |

**Reported accuracy is always strict exact-match** — never the shaped reward. Format compliance hit ~100% within 20 steps and the floor stopped mattering.

## Results

### Training (GSM8K)

| | |
|---|---|
| Before | **8%** (100 held-out test problems, greedy) |
| After 150 steps | **59%** |
| In-batch correct rate | 15% → 43% (first-10 vs last-10 step avg) |
| Data | 512-problem pool, ~1-2 exposures each |
| Cost | 1× Tesla T4, 3.2 h |

Eval questions were verified **disjoint** from training: zero exact-question overlap, and no eval question appears anywhere in the full 7,473-problem training split.

### Transfer experiment — does math help code?

Paired benchmark on MBPP-sanitized (215 problems, pass@1 by **executing** generated code):

| model | pass@1 |
|---|---|
| Qwen3-0.6B-Base | 101/215 = 47.0% |
| Base + math LoRA | 104/215 = 48.4% |

Discordant pairs: 9 base-only, 12 LoRA-only. **Δ = +1.4pp, 95% CI [−2.8, +5.6], exact McNemar p = 0.664.**

**Verdict: no transfer, and no catastrophic forgetting.** The math skill is real but narrow; 150 steps of narrow RL training didn't damage existing abilities.

## Reproduce it

```bash
# training (needs a GPU; Kaggle T4 works)
pip install "transformers>=4.51,<5" peft datasets
python training/train_grpo_math.py        # GRPO_MODE=full|smoke|local

# transfer benchmark (after uploading the adapter as a Kaggle dataset)
python benchmark/bench_code.py
```

`GRPO_MODE=local` runs a 1-step CPU dry run (~8 min) — use it to sanity-check the pipeline before spending GPU quota.

## Honest limitations

- **Partial credit is a crutch** a bigger model/denser data wouldn't need.
- **Training plateaued ~step 76-100** (constant LR, small pool gets re-seen). Final number comes from step-150 weights; `step-100` checkpoint exists and may be better — not separately evaluated.
- **n=100 eval** → ±4-5pp noise. The 8→59 gap is far outside it; smaller comparisons are not.
- **One disclosed hygiene flaw**: the single worked example in prompts turned out to be a GSM8K *test-split* problem. Verified it is **not** in the 100 eval problems, and it appeared identically in before/after prompts — so it cannot explain the delta. Still a process mistake.
- **0/215 debugging saga**: the code benchmark took 5 kernel versions (regex captured prose between fences, missing `def` header, a signature mismatch). Each was caught by re-scoring *saved* generations locally before spending more GPU — see `benchmark/bench_code.py` comments.

## Artifacts

- **Model**: [huggingface.co/Atharva1232/Qwen3-0.6B-GRPO-GSM8K](https://huggingface.co/Atharva1232/Qwen3-0.6B-GRPO-GSM8K)
- **Training kernel**: [kaggle.com/code/atharvagupta123/grpo-math-0-6b](https://www.kaggle.com/code/atharvagupta123/grpo-math-0-6b)
- **Benchmark kernel**: [kaggle.com/code/atharvagupta123/code-bench-base-vs-math-lora](https://www.kaggle.com/code/atharvagupta123/code-bench-base-vs-math-lora)

## License

Code: MIT. Model weights: Apache-2.0 (inherited from [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)). Data: GSM8K (MIT).
