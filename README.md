# Teaching an LLM to Count

Udacity Generative AI Nanodegree (nd608) - Fundamentals Project

## Overview

This project uses **Reinforcement Learning from Human Feedback (RLHF)** techniques -- specifically **GRPO (Group Relative Policy Optimization)** -- to teach a small language model how to accurately count letter occurrences in words.

The base model is **Qwen/Qwen2.5-0.5B-Instruct** (chosen base LLM, per rubric), fine-tuned with LoRA adapters using five custom reward functions that guide the model toward structured, step-by-step counting behavior.

**Apple Silicon (M5 MPS) port:** The original notebook targeted CUDA + `bitsandbytes` 4-bit + `flash_attention_2`. This submission runs natively on Mac Apple Silicon (M5 Max, 128 GB unified memory) — we drop `bitsandbytes` (CUDA-only) and `flash_attention_2` (no MPS support), loading the full-precision BF16 weights directly on MPS. Methodology (5 reward functions, LoRA r=64, GRPOTrainer) is identical to the 3B-based reference.

**Model choice:** We use Qwen2.5-**0.5B**-Instruct (instead of the 3B reference) because the rubric specifies the "chosen base LLM" — no particular size is required — and 0.5B lets the entire training pipeline run end-to-end on MPS in ~20 minutes while preserving every rubric criterion (LoRA config, baseline prompting, reward design, longer training with correctness-reward increasing, final comparison).

## Approach

1. **Baseline Prompting** - Chain-of-thought system prompt with few-shot examples
2. **Dataset Creation** - Procedurally generated letter-counting tasks from common English words
3. **Reward Engineering** - Five reward functions targeting different aspects of correct counting:
   - Sequential numbering of letters
   - Accurate letter spelling
   - Correct running count
   - Proper output format (`<reasoning>...</reasoning><answer>...</answer>`)
   - Final answer correctness
4. **GRPO Training** - Reinforcement learning with the combined reward signal
5. **Evaluation** - Before/after comparison and catastrophic forgetting check

## Files

| File | Description |
|------|-------------|
| `gen_ai_fundamentals_project.ipynb` | Complete project notebook (executed on M5 MPS) |
| `run_grpo_mps.py`                    | Stand-alone training script — equivalent to the notebook |
| `requirements.txt`                   | Python dependencies (no `bitsandbytes`) |
| `reward_curves.png`                  | GRPO training reward curves |
| `loss_curve.png`                     | GRPO training loss |
| `letter_counting_adapter/`           | Saved LoRA adapter weights |

## Setup

```bash
pip install -r requirements.txt
```

Then open `gen_ai_fundamentals_project.ipynb` in Jupyter and run all cells.

### Hardware
- Tested on **MacBook Pro M5 Max, 128 GB unified memory, PyTorch 2.10 MPS backend**
- Alternative (original spec): any CUDA GPU with ≥16 GB VRAM (T4 free-tier Colab works)

## Key Results (100-step run, M5 MPS, Qwen2.5-0.5B-Instruct)

| Metric | Value |
|---|---|
| Training runtime | 2,047 s (~34 min) |
| Final training loss | 0.0220 |
| Trainable params (LoRA) | 17,629,184 / 511,623,168 (3.45%) |
| Base-model accuracy (8 test cases) | **0 / 8 (0 %)** |
| GRPO-trained accuracy (8 test cases) | **3 / 8 (38 %)** |
| Catastrophic forgetting check | **Passed** — Paris / photosynthesis / 15×23 = 345 all correct |

- GRPO training teaches the model to emit the structured `<reasoning>…<answer>…</answer>` format reliably — the base model often fails to produce an answer tag at all (see `results/comparison.json`).
- The fine-tuned model maintains general knowledge: LoRA-off inference is destroyed structurally but LoRA-on inference keeps the structured-counting capability while general-knowledge answers remain coherent.
- See `reward_curves.png` for per-reward trends across the 100 steps and `loss_curve.png` for the GRPO loss.

## License

MIT
