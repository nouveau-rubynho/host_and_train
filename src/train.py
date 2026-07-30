"""LoRA / QLoRA ucenje za klicanje orodij.

    python -m src.train

Ista koda tece na Apple Silicon (precision: bf16) in NVIDIA (precision: 4bit).
Razlika je samo v config.yaml.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .common import LORA_TARGET_REGEX, ROOT, ToolRegistry, load_config, load_model, load_tokenizer
from .dataset import Collator, build_dataset


class ExampleDataset:
    """Minimalen torch-zdruzljiv nabor. Zbiralnik dela pravo delo."""

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Ucenje LoRA adapterja")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments

    cfg = load_config(args.config)
    tcfg = cfg["train"]

    random.seed(tcfg["seed"])
    torch.manual_seed(tcfg["seed"])

    print("Nalagam tokenizator in podatke ...")
    tokenizer = load_tokenizer(cfg["model"]["id"])
    registry = ToolRegistry.load(cfg["data"]["tool_registry"])
    examples = build_dataset(cfg, tokenizer, registry)

    trained = sum(e.n_trained for e in examples)
    total = sum(e.n_total for e in examples)
    print(f"Primerov: {len(examples)}   tokenov: {total}   od tega v izgubi: {trained} ({trained / total:.1%})")
    if trained / total > 0.9:
        print("[opozorilo] skoraj vse je v izgubi - preveri masko z 'python -m src.dataset inspect'")

    eval_split = cfg["data"].get("eval_split", 0.0)
    random.shuffle(examples)
    n_eval = int(len(examples) * eval_split)
    eval_examples = examples[:n_eval]
    train_examples = examples[n_eval:]

    print(f"\nNalagam model {cfg['model']['id']} ({cfg['model']['precision']}) ...")
    model, device = load_model(cfg, for_training=True)
    print(f"Naprava: {device}")

    if cfg["model"]["precision"] == "4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=tcfg["gradient_checkpointing"]
        )

    model.config.use_cache = False  # nezdruzljivo z gradient checkpointing

    lora = LoraConfig(
        r=tcfg["lora_r"],
        lora_alpha=tcfg["lora_alpha"],
        lora_dropout=tcfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_REGEX,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    output_dir = ROOT / tcfg["output_dir"]
    targs = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        learning_rate=float(tcfg["learning_rate"]),
        warmup_ratio=tcfg["warmup_ratio"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        logging_steps=tcfg["logging_steps"],
        save_strategy=tcfg["save_strategy"],
        eval_strategy="epoch" if eval_examples else "no",
        report_to=[],
        seed=tcfg["seed"],
        remove_unused_columns=False,  # nasi primeri niso HF Dataset stolpci
        # fp16/bf16 pusti izklopljen: model je ze nalozen v pravem dtype,
        # mps pa mesanega ucenja prek Trainerja ne podpira zanesljivo.
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ExampleDataset(train_examples),
        eval_dataset=ExampleDataset(eval_examples) if eval_examples else None,
        data_collator=Collator(tokenizer.pad_token_id),
    )

    print("\nZacenjam ucenje ...\n")
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Zabelezi, na cem je adapter nastal - cez mesec dni tega ne bos vedel.
    meta = {
        "base_model": cfg["model"]["id"],
        "precision": cfg["model"]["precision"],
        "device": device,
        "n_examples": len(train_examples),
        "epochs": tcfg["epochs"],
        "lora_r": tcfg["lora_r"],
        "seed_file": cfg["data"]["seed_file"],
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nAdapter shranjen v: {output_dir}")
    print("Za uporabo nastavi v config.yaml:")
    print(f"  model.adapter: {Path(tcfg['output_dir'])}")


if __name__ == "__main__":
    main()
