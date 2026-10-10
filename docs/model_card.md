---
language:
- en
base_model: Qwen/Qwen3-0.6B-Base
tags:
- grpo
- reinforcement-learning
- rlvr
- lora
- math
- qwen3
- from-scratch
- gsm8k
license: apache-2.0
metrics:
- exact_match
---

# Qwen3-0.6B-GRPO-GSM8K

A Qwen3-0.6B base model that learned to solve grade-school math word problems — **without ever being shown a single worked solution.** Trained with GRPO (Group Relative Policy Optimization) written from scratch in ~150 lines of plain PyTorch, no TRL, on one free Kaggle T4 in 3.2 hours.

**GSM8K exact-match: 8% → 59%** (100 held-out test problems, greedy decoding).

---

## The short version

The base model scored **8%** on 100 held-out GSM8K problems. After 150 GRPO steps it scored **59%** — a 7.4× improvement. The model saw no demonstrations, no chain-of-thought targets, no teacher. It got a numeric score for each attempt and that was it.

Here's a correct answer it produced after training:

```
Distance run by Amber: 8 miles.
Distance run by Micah: 8 * 3.5 = 28 miles.
Total distance run by Amber and Micah: 8 + 28 = 36 miles.
Distance run by Ahito: 52 - 36 = 16 miles.
Ahito ran 16 miles.
#### 16
```

That's genuine multi-step arithmetic — multiply, add, subtract from the total — not pattern matching against a memorized answer. The 512 training problems and 100 eval problems were verified disjoint (zero exact-question overlap, and no eval question appears anywhere in the 7,473-problem training split).

## Why this exists

I wanted to see if a tiny model could learn real reasoning from a verifiable reward alone, and how far a from-scratch implementation could get on hardware I actually have.

Two things made this tractable at 0.6B scale:

1. **GRPO drops the critic.** Instead of learning a value function, you sample a *group* of G rollouts per prompt and normalize rewards within the group: `advantage = (reward - mean) / std`. That removes the biggest instability source in PPO at this scale.
2. **The reward is objective.** GSM8K answers are single numbers, so the verifier can't be gamed by a learned reward model. Extract the number after `####`, compare to gold.

The full run: 150 steps × (4 problems × 8 rollouts), LoRA r=16 (10.1M trainable params of 606M), one Tesla T4, **3.2 hours**.

## Does math training transfer to code?

I ran a transfer benchmark on MBPP-sanitized (215 Python problems, pass@1 measured by *executing* the generated code) to test whether math skill generalizes — or whether narrow RL training degrades other abilities.

| model | pass@1 |
|---|---|
| Qwen3-0.6B-Base | 101/215 = 47.0% |
| **Base + this LoRA** | **104/215 = 48.4%** |

Paired analysis: 92 both-pass, 9 base-only, 12 LoRA-only, 102 neither. Delta = **+1.4pp**, 95% CI [−2.8pp, +5.6pp], exact McNemar **p = 0.664**.

**Verdict: no transfer, and no catastrophic forgetting.** The math skill is real but narrow — it doesn't overlap with program synthesis, and critically, 150 steps of narrow RL training didn't damage the model's existing abilities. The low-rank adapter was a gentle touch.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Atharva1232/Qwen3-0.6B-GRPO-GSM8K")
tok = AutoTokenizer.from_pretrained("Atharva1232/Qwen3-0.6B-GRPO-GSM8K")

prompt = (
    "Solve the math problem. Reason step by step. On the LAST line "
    "write the final numeric answer prefixed with ####.\n\n"
    "Problem: {question}\nAnswer:"
)
inputs = tok(prompt.format(question="Natalia sold 48 clips in April..."),
             return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The model is trained to answer in the format `reasoning...\n#### <number>`. Greedy decoding works best.

## Training details

| | |
|---|---|
| Base model | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) |
| Method | GRPO (from scratch, no TRL), LoRA r=16, α=32, all linear layers |
| Trainable params | 10.1M / 606M (1.7%) |
| Data | GSM8K train split, 512-problem pool, ~1-2 exposures each |
| Eval | GSM8K test split, 100 problems, greedy, strict exact-match |
| Group size | 8 rollouts per prompt |
| LR | 5e-5, AdamW (β=0.9/0.95), 5% linear warmup, constant after |
| PPO-style clip | 0.2, 2 gradient epochs per env step |
| Reward | 1.0 exact match; 0.15 format floor + up to 0.5 partial credit for numeric closeness |
| Hardware | 1× Tesla T4 (Kaggle), 3.2 h |
| Frameworks | PyTorch 2.11, Transformers 4.57, PEFT |

### Reward design (the part that mattered)

A cold 0.6B model almost never emits a parseable answer. Worse: if all 8 rollouts in a group get the *same* reward, GRPO's advantage is exactly zero and there's no gradient at all. So the reward is shaped:

- `1.0` — exact match
- `0.15` — produced a parseable `#### <number>` line, wrong value (format floor)
- `+ 0.5·exp(−relative_error)` — continuous credit for numeric closeness

Reported accuracy is **always strict exact-match**, never the shaped reward. Format compliance saturated near 100% within ~20 steps; the shaped floor did its job and then got out of the way.

### Honest limitations

- **The partial-credit shaping is a crutch** that a bigger model or denser data wouldn't need. It buys gradient signal at the cost of some purity.
- **Training plateaued around step 76-100** and drifted slightly down after (no LR decay; the 512-problem pool gets re-seen). The final 59% is from step-150 weights — the best checkpoint may be `step-100`, which I saved but didn't separately evaluate.
- **n=100 eval** has ±4-5pp noise. The 8%→59% gap is far outside it; smaller comparisons are not.
- **One hygiene flaw, disclosed**: the one worked example hardcoded into prompts (Janet's ducks) turned out to be a GSM8K *test-split* problem. It was verified **not** to be in the 100 eval problems, and it appeared identically in both the before- and after-eval prompts — so it cannot explain the delta. It should have been a train-split example; that's a process mistake, not a leak into the metric.

## Training curve

Reward rose steadily, in-batch correct rate went 15% → 43% (first-10 vs last-10 step average), with several perfect groups (8/8 rollouts correct) near the end.

## Files

- `model.safetensors` — merged weights (LoRA applied into base), loads with plain `from_pretrained`
- `tokenizer.*`, `config.json`, `generation_config.json` — from Qwen3-0.6B-Base
- Adapter-only version + all checkpoints (step 50/100/150): see the linked Kaggle artifacts

## Citation

```bibtex
@misc{atharvagupta123-grpo-gsm8k-0.6b,
  title = {Qwen3-0.6B-GRPO-GSM8K: GSM8K 8% to 59% with from-scratch GRPO on one T4},
  author = {Atharva1232},
  year = {2026},
  howpublished = {\url{https://huggingface.co/Atharva1232/Qwen3-0.6B-GRPO-GSM8K}}
}
```
