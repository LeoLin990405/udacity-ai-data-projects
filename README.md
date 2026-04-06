# Teaching an LLM to Count

Udacity Generative AI Nanodegree - Fundamentals Project

## Overview

This project uses **Reinforcement Learning from Human Feedback (RLHF)** techniques -- specifically **GRPO (Group Relative Policy Optimization)** -- to teach a small language model how to accurately count letter occurrences in words.

The base model is **Qwen/Qwen2.5-3B-Instruct**, fine-tuned with LoRA adapters using five custom reward functions that guide the model toward structured, step-by-step counting behavior.

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
| `gen_ai_fundamentals_project.ipynb` | Complete project notebook |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

Then open `gen_ai_fundamentals_project.ipynb` in Jupyter or Google Colab and run all cells.

**Note:** Training requires a GPU with at least 16GB VRAM. A T4 or better on Google Colab (free tier) works.

## Key Results

- The model learns structured letter-by-letter counting through GRPO training
- Reward curves show steady improvement across all five reward dimensions
- The fine-tuned model maintains general knowledge (no catastrophic forgetting)

## License

MIT
