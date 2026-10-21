"""Separate comparison: base vs GRPO-trained Qwen2.5-0.5B on letter counting.

Runs after training. Uses peft `disable_adapter()` to toggle LoRA on/off on the
same model instance (avoids the MPS 4 GB NDArray limit hit when loading two
independent copies of the base model).

Also runs catastrophic-forgetting check with general-knowledge questions.

Output:
  results/comparison.json
  results/general_knowledge.json
"""
import json
import os
import re
import sys
import time


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


def main():
    sys.stdout.reconfigure(line_buffering=True)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    ADAPTER_PATH = "./letter_counting_adapter"
    # MPS hits `NDArray > 2**32` assertion on Qwen+LoRA generate path; use CPU
    # for the short comparison phase (8 examples × ~200 tokens is <3 min).
    device = os.environ.get("COMPARE_DEVICE", "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] device={device}", flush=True)

    os.makedirs("results", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[{time.strftime('%H:%M:%S')}] loading base model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device)
    print(f"[{time.strftime('%H:%M:%S')}] attaching LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    def _extract(text):
        rm = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        am = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        return (rm.group(1).strip() if rm else "", am.group(1).strip() if am else "")

    def generate(prompt, max_new=220, system=SYSTEM_PROMPT):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    cases = [
        ("strawberry", "r", 3),
        ("mississippi", "s", 4),
        ("banana", "a", 3),
        ("bookkeeper", "e", 2),
        ("communication", "m", 2),
        ("butterfly", "t", 2),
        ("chocolate", "c", 2),
        ("exaggerate", "g", 2),
    ]

    print(f"\n[{time.strftime('%H:%M:%S')}] === COMPARISON: Base (LoRA off) vs Trained (LoRA on) ===", flush=True)
    comparison = []
    base_correct = trained_correct = 0
    for word, letter, expected in cases:
        q = f"How many times does the letter '{letter}' appear in the word '{word}'?"
        with model.disable_adapter():
            b_resp = generate(q)
        _, b_ans = _extract(b_resp)
        t_resp = generate(q)
        _, t_ans = _extract(t_resp)
        b_ok = b_ans == str(expected)
        t_ok = t_ans == str(expected)
        base_correct += b_ok
        trained_correct += t_ok
        comparison.append({
            "word": word, "letter": letter, "expected": expected,
            "base_answer": b_ans, "trained_answer": t_ans,
            "base_response": b_resp, "trained_response": t_resp,
            "base_ok": b_ok, "trained_ok": t_ok,
        })
        print(f"  '{letter}' in '{word}' exp={expected} | base={b_ans}({'OK' if b_ok else 'X '}) trained={t_ans}({'OK' if t_ok else 'X '})", flush=True)

    print(f"\n  BASE:    {base_correct}/{len(cases)} ({100*base_correct/len(cases):.0f}%)")
    print(f"  TRAINED: {trained_correct}/{len(cases)} ({100*trained_correct/len(cases):.0f}%)")

    with open("results/comparison.json", "w") as f:
        json.dump({
            "cases": comparison,
            "base_correct": base_correct,
            "trained_correct": trained_correct,
            "total": len(cases),
        }, f, indent=2)

    # Catastrophic forgetting check
    print(f"\n[{time.strftime('%H:%M:%S')}] === CATASTROPHIC FORGETTING CHECK ===", flush=True)
    general_q = [
        "What is the capital of France?",
        "Explain photosynthesis in one sentence.",
        "What is 15 * 23?",
    ]
    general_sys = "You are a helpful assistant. Answer questions concisely."
    general_results = []
    for q in general_q:
        resp = generate(q, max_new=150, system=general_sys).strip()
        general_results.append({"q": q, "a": resp})
        print(f"  Q: {q}\n  A: {resp[:250]}\n", flush=True)

    with open("results/general_knowledge.json", "w") as f:
        json.dump(general_results, f, indent=2)

    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
