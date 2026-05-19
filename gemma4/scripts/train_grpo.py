"""GRPO RL pass over the v3 SFT checkpoint.

Goals (success targets vs v3 baseline on golden_en re-eval):
  tool_call rate ≥ 0.30 (vs 0.02), format_ok ≥ 0.80 (vs 0.52),
  byte_cap ≥ 0.80 (vs 0.48), response_quality ≥ 0.65 (vs 0.506).

Recipe (consistent with v3 SFT in train_sft.py):
  - Unsloth FastModel from out/lastbox-gemma4-e2b-sft-v3-nothink/ (HF dir)
  - PEFT LoRA r=8, alpha=8, same target modules as SFT
  - TRL GRPOTrainer with reward = gemma4.grpo.reward.reward_func
  - num_generations=4 (8 would be nicer but doubles GB10 time)
  - lr=5e-6, beta=0.04, 1 epoch over train_v2
  - bf16 on GB10, no 4-bit (full activations)
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TOOL_CALL_RE = re.compile(r"<tool_call>\s*\{\s*\"name\"\s*:\s*\"([^\"]+)\"", re.DOTALL)


def first_expected_tool(messages: list[dict]) -> str | None:
    """Pull the first assistant tool_call's name, if any."""
    for m in messages:
        if m["role"] == "assistant":
            mt = TOOL_CALL_RE.search(m.get("content", ""))
            return mt.group(1) if mt else None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                    default=Path("out/lastbox-gemma4-e2b-sft-v3-nothink"))
    ap.add_argument("--train", type=Path,
                    default=Path("gemma4/data/train_v2.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("out/lastbox-gemma4-e2b-grpo-v4"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-prompt", type=int, default=1536)
    ap.add_argument("--max-completion", type=int, default=256)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="0 = all; smaller for smoke runs")
    ap.add_argument("--export-gguf", action="store_true", default=True)
    ap.add_argument("--no-export-gguf", action="store_false", dest="export_gguf")
    ap.add_argument("--quant", type=str, default="q4_k_m")
    args = ap.parse_args()

    print(f"[grpo] base SFT checkpoint: {args.base}", flush=True)
    print(f"[grpo] train: {args.train}", flush=True)
    print(f"[grpo] out: {args.out}", flush=True)

    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    from datasets import Dataset
    from trl import GRPOTrainer, GRPOConfig
    import torch

    from gemma4.grpo.reward import reward_func

    # ─── Load base SFT checkpoint ───────────────────────────────────────
    t0 = time.time()
    base_id = str(args.base)
    if (args.base / "lora").exists() and not (args.base / "adapter_config.json").exists():
        # Path layout has lora/ subdir — feed that to from_pretrained
        merged_dir = args.base / "merged-for-grpo"
        if not merged_dir.exists():
            print(f"[grpo] base is a LoRA-only checkpoint; loading via PEFT merge",
                  flush=True)
    model, processor = FastModel.from_pretrained(
        model_name=base_id,
        dtype=None,
        max_seq_length=args.max_prompt + args.max_completion,
        load_in_4bit=False,
        full_finetuning=False,
    )
    processor = get_chat_template(processor, chat_template="gemma-4")
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(processor, "chat_template", None) and not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = processor.chat_template
    print(f"[grpo] model + tokenizer loaded in {time.time()-t0:.1f}s", flush=True)

    # Re-attach a LoRA on top of the SFT-trained weights — GRPO updates this
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=42,
    )

    # ─── Build prompt-only dataset with reward kwargs ──────────────────
    rows = []
    with args.train.open() as f:
        for line in f:
            d = json.loads(line)
            msgs = d.get("messages", [])
            if not msgs:
                continue
            # Use the system+first-user turn as the prompt
            prompt_msgs = []
            for m in msgs:
                if m["role"] in ("system", "user"):
                    prompt_msgs.append({"role": m["role"], "content": m["content"]})
                if m["role"] == "user":
                    break  # stop at first user turn
            if not prompt_msgs:
                continue
            try:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                continue
            rows.append({
                "prompt": prompt_text,
                "expected_tool": first_expected_tool(msgs) or "",
                "source": d.get("source") or "",
            })
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    ds = Dataset.from_list(rows)
    print(f"[grpo] dataset: {len(ds)} prompts; "
          f"with-tool: {sum(1 for r in rows if r['expected_tool'])}", flush=True)

    # ─── GRPO config & training ────────────────────────────────────────
    cfg = GRPOConfig(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        logging_steps=2,
        save_steps=200,
        save_total_limit=2,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion,
        beta=args.beta,
        temperature=0.7,
        top_p=0.9,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        reward_funcs=reward_func,
        args=cfg,
    )
    t1 = time.time()
    trainer.train()
    train_secs = time.time() - t1
    print(f"[grpo] training done in {train_secs:.0f}s", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out / "lora"))
    processor.save_pretrained(str(args.out / "lora"))
    print(f"[grpo] LoRA saved to {args.out / 'lora'}", flush=True)

    if args.export_gguf:
        t2 = time.time()
        try:
            model.save_pretrained_gguf(
                str(args.out), processor, quantization_method=args.quant,
            )
            print(f"[grpo] GGUF ({args.quant}) exported in {time.time()-t2:.0f}s",
                  flush=True)
        except Exception as e:
            print(f"[grpo] GGUF export failed: {e}", flush=True)

    meta = {
        "phase": "grpo",
        "base": str(args.base),
        "epochs": args.epochs,
        "lr": args.lr,
        "num_generations": args.num_generations,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_samples": len(ds),
        "train_seconds": train_secs,
        "quant": args.quant if args.export_gguf else None,
        "reward": "0.5*tool_match + 0.3*format_ok + 0.2*byte_cap_ok",
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[grpo] done → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
