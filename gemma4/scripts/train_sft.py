"""SFT LoRA Gemma 4 E2B-it dla LastBox (hackathon Gemma 4 Good).

Recipe oparte na oficjalnej Unsloth Gemma 4 doc:
- FastModel (NIE FastLanguageModel)
- chat_template = "gemma-4-thinking"
- LoRA r=8 alpha=8 (małe! E2B się przeucza przy r=32)
- finetune_vision_layers=False (text-only PL)
- load_in_4bit=True, full_finetuning=False

Trening idzie na GB10 (aarch64, CUDA 13, 121GB RAM) — bf16, batch 4 x grad_accum 4.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="unsloth/gemma-4-E2B-it")
    ap.add_argument("--train", type=Path, default=Path("gemma4/data/train.jsonl"))
    ap.add_argument("--val", type=Path, default=Path("gemma4/data/val.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("out/lastbox-gemma4-e2b-sft"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--load-in-4bit", action="store_true", default=False,
                    help="QLoRA 4-bit (default OFF: GB10 ma 121GB, bf16 LoRA wystarcza)")
    ap.add_argument("--chat-template", type=str, default="gemma-4-thinking",
                    choices=["gemma-4", "gemma-4-thinking"],
                    help="Unsloth chat template — 'gemma-4' suppresses CoT, 'gemma-4-thinking' enables it")
    ap.add_argument("--export-gguf", action="store_true", default=True)
    ap.add_argument("--no-export-gguf", action="store_false", dest="export_gguf")
    ap.add_argument("--quant", type=str, default="q4_k_m",
                    choices=["q4_k_m", "q5_k_m", "q8_0", "f16"])
    args = ap.parse_args()

    print(f"[INFO] Base: {args.base}")
    print(f"[INFO] Train: {args.train}  Val: {args.val}")
    print(f"[INFO] LoRA r={args.lora_r} alpha={args.lora_alpha}  epochs={args.epochs} lr={args.lr}")

    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch

    t0 = time.time()
    model, processor = FastModel.from_pretrained(
        model_name=args.base,
        dtype=None,
        max_seq_length=args.max_seq,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )
    processor = get_chat_template(processor, chat_template=args.chat_template)
    # Gemma 4 to multimodal model — processor wraps text tokenizer + image processor.
    # Do treningu text-only używamy wewnętrznego GemmaTokenizer'a (is_fast=True).
    # Chat template z processora propaguje się na inner tokenizer.
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(processor, "chat_template", None) and not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = processor.chat_template
    print(f"[INFO] Model + tokenizer załadowane w {time.time()-t0:.1f}s")
    print(f"[INFO] Processor: {type(processor).__name__}  Tokenizer: {type(tokenizer).__name__} fast={getattr(tokenizer, 'is_fast', None)}")

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

    data_files = {"train": str(args.train)}
    if args.val.exists():
        data_files["validation"] = str(args.val)
    raw_ds = load_dataset("json", data_files=data_files)

    # Pre-tokenize ręcznie text-only tokenizerem. SFTTrainer + Unsloth's
    # _unsloth_get_batch_samples nie dziedziczy auto-tokenizacji z multimodal
    # Processora — musimy dać input_ids/labels/attention_mask gotowe.
    def _tok(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        enc = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq,
            padding=False,
            return_tensors=None,
        )
        enc["labels"] = list(enc["input_ids"])
        return enc

    cols_to_drop = list(raw_ds["train"].column_names)
    ds = raw_ds.map(_tok, batched=False, remove_columns=cols_to_drop)
    ds = ds.filter(lambda ex: len(ex.get("input_ids", [])) > 4)

    training_args = SFTConfig(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        eval_strategy="steps" if "validation" in ds else "no",
        eval_steps=50 if "validation" in ds else None,
        optim="adamw_8bit" if args.load_in_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=42,
        max_length=args.max_seq,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        args=training_args,
    )
    t1 = time.time()
    trainer.train()
    train_secs = time.time() - t1
    print(f"[INFO] Trening zajął {train_secs:.0f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out / "lora"))
    processor.save_pretrained(str(args.out / "lora"))
    print(f"[OK] LoRA zapisany w {args.out / 'lora'}")

    if args.export_gguf:
        t2 = time.time()
        try:
            model.save_pretrained_gguf(
                str(args.out),
                processor,
                quantization_method=args.quant,
            )
            print(f"[OK] GGUF ({args.quant}) zapisany w {time.time()-t2:.0f}s")
        except Exception as e:
            print(f"[WARN] GGUF export failed: {e}. Zrobimy ręcznie potem.")

    meta = {
        "phase": "sft",
        "base": args.base,
        "chat_template": args.chat_template,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "max_seq": args.max_seq,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "train_samples": len(ds["train"]),
        "val_samples": len(ds.get("validation", [])),
        "train_seconds": train_secs,
        "quant": args.quant if args.export_gguf else None,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[DONE] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
