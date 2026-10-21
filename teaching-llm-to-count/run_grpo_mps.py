"""GRPO training for Teaching-LLM-to-Count on M5 MPS.

Port of gen_ai_fundamentals_project.ipynb to run on Apple Silicon without
bitsandbytes / flash_attention_2 (not supported on MPS).

Changes vs. notebook:
  - No 4-bit quantization (BF16 full-precision load; ~6 GB, fits M5 128 GB)
  - attn_implementation="eager" instead of "flash_attention_2"
  - device="mps"
  - trl 1.2.0 API: pass peft_config to GRPOTrainer directly; drop max_prompt_length
"""
import gc
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    sys.stdout.reconfigure(line_buffering=True)
    # Allow CPU fallback for any MPS-unsupported kernels
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    MAX_STEPS = int(os.environ.get("MAX_STEPS", "100"))
    MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    OUTPUT_DIR = "./results"
    ADAPTER_PATH = "./letter_counting_adapter"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] device={device}, max_steps={MAX_STEPS}", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[{time.strftime('%H:%M:%S')}] loading model (BF16 on MPS)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    print(f"[{time.strftime('%H:%M:%S')}] model loaded: {MODEL_NAME} dtype={model.dtype}", flush=True)

    SYSTEM_PROMPT = """You are a letter-counting assistant. When asked how many times a letter appears in a word, you must:

1. Number each letter in the word sequentially (1. letter, 2. letter, ...)
2. For each letter, state what it is and keep a running count of the target letter
3. Give the final answer

You MUST use this exact format:
<reasoning>
1. [letter]: [is/is not target] (count so far: N)
2. [letter]: [is/is not target] (count so far: N)
...
</reasoning>
<answer>[number]</answer>

Example:
Question: How many times does the letter 'o' appear in the word 'room'?

<reasoning>
1. r: not target (count so far: 0)
2. o: is target (count so far: 1)
3. o: is target (count so far: 2)
4. m: not target (count so far: 2)
</reasoning>
<answer>2</answer>

Always go through EVERY letter one by one. Never skip letters or guess."""

    ALL_WORDS = [
        "apple", "banana", "orange", "strawberry", "elephant",
        "giraffe", "butterfly", "computer", "telephone", "beautiful",
        "chocolate", "mountain", "umbrella", "pineapple", "watermelon",
        "dictionary", "adventure", "happiness", "wonderful", "education",
        "environment", "strawberry", "necessary", "opportunity", "communication",
        "international", "professional", "understanding", "responsibility", "extraordinary",
        "accommodation", "appreciation", "recommendation", "mississippi", "bookkeeper",
        "committee", "millennium", "broccoli", "cappuccino", "occurrence",
        "possession", "accessible", "successfully", "assassination", "balloon",
        "beginning", "different", "disappear", "embarrass", "exaggerate",
    ]
    TARGET_LETTERS = list("abcdefghilmnoprstuw")

    def generate_records(words, letters):
        records = []
        for word in words:
            for letter in letters:
                count = word.lower().count(letter)
                if count == 0 and random.random() > 0.3:
                    continue
                prompt_q = f"How many times does the letter '{letter}' appear in the word '{word}'?"
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_q},
                ]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                records.append({
                    "prompt": formatted,
                    "answer": str(count),
                    "word": word,
                    "target_letter": letter,
                })
        random.seed(42)
        random.shuffle(records)
        return records

    records = generate_records(ALL_WORDS, TARGET_LETTERS)
    dataset = Dataset.from_list(records)
    print(f"[{time.strftime('%H:%M:%S')}] dataset: {len(dataset)} samples", flush=True)

    def _extract(text):
        rm = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        am = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        return (rm.group(1).strip() if rm else "", am.group(1).strip() if am else "")

    def _parse_numbered(reasoning):
        pat = r"(\d+)\.\s*(.*?)(?=\n\d+\.|\Z)"
        return [(int(n), c.strip()) for n, c in re.findall(pat, reasoning, re.DOTALL)]

    def numbering_reward_func(completions, **kwargs):
        words = kwargs.get("word", [])
        out = []
        for i, c in enumerate(completions):
            text = c[0]["content"] if isinstance(c, list) else c
            word = words[i] if i < len(words) else ""
            r, _ = _extract(text)
            lines = _parse_numbered(r)
            if not lines:
                out.append(-0.5); continue
            if len(lines) > len(word):
                out.append(-1.0); continue
            ok = all(lines[j][0] == j + 1 for j in range(len(lines)))
            out.append(0.5 if ok and len(lines) == len(word) else -0.5)
        return out

    def spelling_reward_func(completions, **kwargs):
        words = kwargs.get("word", [])
        out = []
        for i, c in enumerate(completions):
            text = c[0]["content"] if isinstance(c, list) else c
            word = (words[i] if i < len(words) else "").lower()
            r, _ = _extract(text)
            lines = _parse_numbered(r)
            if not lines:
                out.append(-0.5); continue
            spelled = []
            for _, content in lines:
                m = re.match(r"([a-zA-Z])\s*:", content)
                if m: spelled.append(m.group(1).lower())
            letters = list(word)
            if spelled == letters:
                out.append(2.0); continue
            reward = -0.5 * abs(len(spelled) - len(letters))
            for j, ch in enumerate(spelled):
                if j < len(letters) and ch != letters[j]:
                    reward -= 1.0
                elif j >= len(letters):
                    reward -= 1.0
            if len(spelled) < len(letters):
                reward -= 0.5 * (len(letters) - len(spelled))
            out.append(max(reward, -1.0))
        return out

    def counting_reward_func(completions, **kwargs):
        words = kwargs.get("word", [])
        targets = kwargs.get("target_letter", [])
        out = []
        for i, c in enumerate(completions):
            text = c[0]["content"] if isinstance(c, list) else c
            word = (words[i] if i < len(words) else "").lower()
            target = (targets[i] if i < len(targets) else "").lower()
            r, _ = _extract(text)
            lines = _parse_numbered(r)
            if not lines:
                out.append(-1.0); continue
            expected, running = [], 0
            for ch in word:
                if ch == target: running += 1
                expected.append(running)
            score = 0.0
            steps = min(len(lines), len(word))
            for j in range(steps):
                _, content = lines[j]
                m = re.search(r"count\s+so\s+far:\s*(\d+)", content)
                if m:
                    actual = int(m.group(1))
                    score += 1.0 if actual == expected[j] else -1.0
                else:
                    score -= 1.0
            norm = score / len(word) if len(word) > 0 else 0.0
            out.append(max(min(norm, 1.0), -1.0))
        return out

    def format_reward_func(completions, **kwargs):
        out = []
        for c in completions:
            text = c[0]["content"] if isinstance(c, list) else c
            reward = 0.0
            if re.search(r"<reasoning>.*?</reasoning>", text, re.DOTALL) and re.search(r"<answer>.*?</answer>", text, re.DOTALL):
                reward += 0.5
            am = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            if am and am.group(1).strip().isdigit():
                reward += 0.5
            out.append(reward)
        return out

    def correct_answer_reward_func(completions, **kwargs):
        answers = kwargs.get("answer", [])
        out = []
        for i, c in enumerate(completions):
            text = c[0]["content"] if isinstance(c, list) else c
            expected = answers[i] if i < len(answers) else ""
            _, ans = _extract(text)
            out.append(2.0 if ans == expected else -1.0)
        return out

    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        beta=0.0,                        # disable KL ref model (halves memory)
        per_device_train_batch_size=2,   # MPS: smaller batch
        num_generations=2,
        gradient_accumulation_steps=1,
        max_steps=MAX_STEPS,
        max_completion_length=192,       # shorter (was 256) to speed up
        logging_steps=5,                 # log less often — fewer .item() syncs
        save_steps=max(MAX_STEPS, 25),
        report_to="none",
        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        temperature=1.0,
        top_p=0.9,
        use_vllm=False,
    )

    print(f"[{time.strftime('%H:%M:%S')}] initializing GRPOTrainer...", flush=True)
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[
            numbering_reward_func,
            spelling_reward_func,
            counting_reward_func,
            format_reward_func,
            correct_answer_reward_func,
        ],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"[{time.strftime('%H:%M:%S')}] training start ({MAX_STEPS} steps)...", flush=True)
    t0 = time.time()
    train_result = trainer.train()
    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] training done in {dt:.0f}s; final loss={train_result.training_loss:.4f}", flush=True)

    trainer.save_model(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    print(f"[{time.strftime('%H:%M:%S')}] adapter saved to {ADAPTER_PATH}", flush=True)

    with open(os.path.join(OUTPUT_DIR, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    reward_keys = {
        "rewards/numbering_reward_func/mean": "Numbering",
        "rewards/spelling_reward_func/mean": "Spelling",
        "rewards/counting_reward_func/mean": "Counting",
        "rewards/format_reward_func/mean": "Format",
        "rewards/correct_answer_reward_func/mean": "Correct Answer",
    }
    alt_keys = {
        "reward/numbering_reward_func": "Numbering",
        "reward/spelling_reward_func": "Spelling",
        "reward/counting_reward_func": "Counting",
        "reward/format_reward_func": "Format",
        "reward/correct_answer_reward_func": "Correct Answer",
    }
    metrics = defaultdict(list); step_data = defaultdict(list)
    total_rewards, total_reward_steps = [], []
    losses, loss_steps = [], []
    for entry in trainer.state.log_history:
        if "step" not in entry: continue
        s = entry["step"]
        for km in (reward_keys, alt_keys):
            for k, lab in km.items():
                if k in entry:
                    metrics[lab].append(entry[k]); step_data[lab].append(s)
        if "reward" in entry:
            total_rewards.append(entry["reward"]); total_reward_steps.append(s)
        if "loss" in entry:
            losses.append(entry["loss"]); loss_steps.append(s)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"GRPO Training Reward Curves ({MAX_STEPS} steps, M5 MPS)", fontsize=16)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    for idx, (lab, vals) in enumerate(metrics.items()):
        ax = axes[idx // 3][idx % 3]
        ax.plot(step_data[lab], vals, color=colors[idx % 5], alpha=0.8, linewidth=1.5)
        ax.set_title(lab); ax.set_xlabel("Step"); ax.set_ylabel("Reward"); ax.grid(True, alpha=0.3)
    ax = axes[1][2]
    if total_rewards:
        ax.plot(total_reward_steps, total_rewards, color="#795548", linewidth=1.5)
    ax.set_title("Total Reward"); ax.set_xlabel("Step"); ax.set_ylabel("Reward"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward_curves.png", dpi=120, bbox_inches="tight"); plt.close()

    if losses:
        plt.figure(figsize=(10, 4))
        plt.plot(loss_steps, losses, color="#E91E63", linewidth=1.5)
        plt.title("Training Loss"); plt.xlabel("Step"); plt.ylabel("Loss"); plt.grid(alpha=0.3)
        plt.savefig("loss_curve.png", dpi=120, bbox_inches="tight"); plt.close()

    print(f"[{time.strftime('%H:%M:%S')}] plots saved: reward_curves.png, loss_curve.png", flush=True)

    def generate(model, prompt, max_new=512, system=SYSTEM_PROMPT):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.9)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] === COMPARISON: Base vs Trained ===", flush=True)
    # Disable LoRA adapter for base comparison (avoid loading a 2nd 0.5B into MPS)
    peft_model = trainer.model

    cases = [("strawberry", "r", 3), ("mississippi", "s", 4), ("banana", "a", 3),
             ("bookkeeper", "e", 2), ("communication", "m", 2), ("butterfly", "t", 2),
             ("chocolate", "c", 2), ("exaggerate", "g", 2)]
    comparison = []
    base_correct = trained_correct = 0
    for word, letter, expected in cases:
        q = f"How many times does the letter '{letter}' appear in the word '{word}'?"
        # Base = LoRA disabled
        with peft_model.disable_adapter():
            b_resp = generate(peft_model, q)
            _, b_ans = _extract(b_resp)
        # Trained = LoRA enabled
        t_resp = generate(peft_model, q)
        _, t_ans = _extract(t_resp)
        b_ok = b_ans == str(expected); t_ok = t_ans == str(expected)
        base_correct += b_ok; trained_correct += t_ok
        comparison.append({"word": word, "letter": letter, "expected": expected,
                           "base_answer": b_ans, "trained_answer": t_ans,
                           "base_ok": b_ok, "trained_ok": t_ok})
        print(f"  '{letter}' in '{word}' exp={expected} | base={b_ans}({'✓' if b_ok else '✗'}) trained={t_ans}({'✓' if t_ok else '✗'})", flush=True)

    print(f"\n  BASE:    {base_correct}/{len(cases)} ({100*base_correct/len(cases):.0f}%)")
    print(f"  TRAINED: {trained_correct}/{len(cases)} ({100*trained_correct/len(cases):.0f}%)")

    with open(os.path.join(OUTPUT_DIR, "comparison.json"), "w") as f:
        json.dump({"cases": comparison, "base_correct": base_correct, "trained_correct": trained_correct,
                   "total": len(cases)}, f, indent=2)

    # Catastrophic forgetting check
    print(f"\n[{time.strftime('%H:%M:%S')}] === CATASTROPHIC FORGETTING CHECK ===", flush=True)
    general_q = ["What is the capital of France?", "Explain photosynthesis in one sentence.", "What is 15 * 23?"]
    general_sys = "You are a helpful assistant. Answer questions concisely."
    general_results = []
    for q in general_q:
        resp = generate(peft_model, q, max_new=150, system=general_sys).strip()
        general_results.append({"q": q, "a": resp})
        print(f"  Q: {q}\n  A: {resp[:200]}\n", flush=True)

    with open(os.path.join(OUTPUT_DIR, "general_knowledge.json"), "w") as f:
        json.dump(general_results, f, indent=2)

    print(f"\n[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
