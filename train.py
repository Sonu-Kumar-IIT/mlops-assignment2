"""
train.py — Load pre-processed data, fine-tune DistilBERT, and save the model.

Usage:
    python train.py
    python train.py --epochs 5 --batch_size 16 --learning_rate 3e-5
    python train.py --wandb --wandb_project mlops-assignment2 --wandb_run distilbert-run-1
    python train.py --data_dir data --model_save_dir distilbert-reviews-genres

Prerequisites:
    Run data.py first to generate the artefacts in `data_dir/`.

What this script does:
  1. Loads tokenizer encodings and encoded labels from `data_dir/`.
  2. Loads the label maps from `data_dir/label_maps.json`.
  3. Optionally initialises a W&B run if --wandb is passed.
  4. Loads DistilBertForSequenceClassification from HuggingFace Hub.
  5. Configures TrainingArguments and creates a Trainer.
  6. Runs trainer.train() and saves the fine-tuned model + tokenizer.
"""

import argparse
import json
import os
import pickle
import sys

import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from utils import (
    MODEL_NAME,
    MODEL_SAVE_DIR,
    DATA_DIR,
    DEVICE_NAME,
    MyDataset,
    compute_metrics,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_artefacts(data_dir: str):
    """Load pickled encodings and label lists produced by data.py."""
    required = [
        "train_encodings.pkl",
        "test_encodings.pkl",
        "train_labels_encoded.pkl",
        "test_labels_encoded.pkl",
    ]
    for fname in required:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(
                f"[train.py] ERROR: Required artefact not found: {path}\n"
                f"           Run data.py first to generate preprocessed data.",
                file=sys.stderr,
            )
            sys.exit(1)

    def load(fname):
        with open(os.path.join(data_dir, fname), "rb") as fh:
            return pickle.load(fh)

    return (
        load("train_encodings.pkl"),
        load("test_encodings.pkl"),
        load("train_labels_encoded.pkl"),
        load("test_labels_encoded.pkl"),
    )


def load_label_maps(data_dir: str):
    """Load label2id / id2label from the JSON file written by data.py."""
    path = os.path.join(data_dir, "label_maps.json")
    if not os.path.exists(path):
        print(
            f"[train.py] ERROR: label_maps.json not found in '{data_dir}'.\n"
            "           Run data.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        maps = json.load(fh)
    label2id = maps["label2id"]
    id2label = {int(k): v for k, v in maps["id2label"].items()}
    return label2id, id2label


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    print(f"\n[train.py] Using device: {DEVICE_NAME}")

    # ── 1. Load artefacts ──────────────────────────────────────────────────────
    print(f"[train.py] Loading preprocessed data from '{args.data_dir}/' ...")
    train_enc, test_enc, train_labels, test_labels = load_artefacts(args.data_dir)
    label2id, id2label = load_label_maps(args.data_dir)
    num_labels = len(label2id)
    print(f"  {len(train_labels)} training examples | {len(test_labels)} test examples")
    print(f"  {num_labels} classes: {list(label2id.keys())}")

    # ── 2. Build PyTorch datasets ──────────────────────────────────────────────
    train_dataset = MyDataset(train_enc, train_labels)
    test_dataset  = MyDataset(test_enc,  test_labels)

    # ── 3. Optional W&B initialisation ────────────────────────────────────────
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run,
                config={
                    "model":         args.model_name,
                    "epochs":        args.epochs,
                    "batch_size":    args.batch_size,
                    "learning_rate": args.learning_rate,
                    "max_length":    args.max_length,
                    "dataset":       "UCSD Goodreads",
                    "platform":      "local",
                },
            )
            report_to = "wandb"
            print(f"[train.py] W&B run '{args.wandb_run}' initialised "
                  f"(project: {args.wandb_project}).")
        except ImportError:
            print("[train.py] [WARNING] wandb not installed. "
                  "Install with: pip install wandb\nContinuing without W&B logging.",
                  file=sys.stderr)
            report_to = "none"
    else:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"

    # ── 4. Load pre-trained model ──────────────────────────────────────────────
    print(f"\n[train.py] Loading model '{args.model_name}' ({num_labels} output labels) ...")
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE_NAME)

    # ── 5. TrainingArguments ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=report_to,
        run_name=args.wandb_run if args.wandb else None,
        fp16=args.fp16 and DEVICE_NAME == "cuda",   # Only enable FP16 on GPU
    )

    print(f"[train.py] TrainingArguments:")
    print(f"  epochs={args.epochs} | train_batch={args.batch_size} | "
          f"eval_batch={args.eval_batch_size} | lr={args.learning_rate} | "
          f"fp16={'yes' if training_args.fp16 else 'no'}")

    # ── 6. Trainer ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # ── 7. Train ───────────────────────────────────────────────────────────────
    print(f"\n[train.py] Starting training ...")
    trainer.train()

    # ── 8. Save model and tokenizer ───────────────────────────────────────────
    os.makedirs(args.model_save_dir, exist_ok=True)
    trainer.save_model(args.model_save_dir)

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    tokenizer.save_pretrained(args.model_save_dir)

    print(f"\n[train.py] Model and tokenizer saved to '{args.model_save_dir}/'")

    # ── 9. Log HF URL to W&B (if applicable) ──────────────────────────────────
    if args.wandb and args.hf_model_url:
        try:
            import wandb
            wandb.run.summary["huggingface_model"] = args.hf_model_url
            print(f"[train.py] HF model URL logged to W&B: {args.hf_model_url}")
        except Exception:
            pass

    # ── 10. Close W&B run ─────────────────────────────────────────────────────
    if args.wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    print("\n[train.py] Training complete. Run eval.py next.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on Goodreads genre classification."
    )

    # Paths
    parser.add_argument("--data_dir",       type=str, default=DATA_DIR,
                        help=f"Directory containing artefacts from data.py (default: {DATA_DIR}).")
    parser.add_argument("--output_dir",     type=str, default="./results",
                        help="Intermediate checkpoint directory (default: ./results).")
    parser.add_argument("--model_save_dir", type=str, default=MODEL_SAVE_DIR,
                        help=f"Directory to save the final fine-tuned model (default: {MODEL_SAVE_DIR}).")
    parser.add_argument("--model_name",     type=str, default=MODEL_NAME,
                        help=f"HuggingFace model identifier (default: {MODEL_NAME}).")
    parser.add_argument("--max_length",     type=int, default=512,
                        help="Max token length used during data prep (informational, default: 512).")

    # Training hyperparameters
    parser.add_argument("--epochs",          type=int,   default=3,
                        help="Number of training epochs (default: 3).")
    parser.add_argument("--batch_size",      type=int,   default=10,
                        help="Per-device training batch size (default: 10).")
    parser.add_argument("--eval_batch_size", type=int,   default=16,
                        help="Per-device evaluation batch size (default: 16).")
    parser.add_argument("--learning_rate",   type=float, default=5e-5,
                        help="Initial learning rate (default: 5e-5).")
    parser.add_argument("--warmup_steps",    type=int,   default=100,
                        help="LR scheduler warmup steps (default: 100).")
    parser.add_argument("--weight_decay",    type=float, default=0.01,
                        help="Weight decay coefficient (default: 0.01).")
    parser.add_argument("--logging_steps",   type=int,   default=100,
                        help="Log training loss every N steps (default: 100).")
    parser.add_argument("--fp16",            action="store_true",
                        help="Enable FP16 mixed-precision training (GPU only).")

    # W&B
    parser.add_argument("--wandb",          action="store_true",
                        help="Enable Weights & Biases experiment tracking.")
    parser.add_argument("--wandb_project",  type=str, default="mlops-assignment2",
                        help="W&B project name (default: mlops-assignment2).")
    parser.add_argument("--wandb_run",      type=str, default="distilbert-run-1",
                        help="W&B run name (default: distilbert-run-1).")
    parser.add_argument("--hf_model_url",   type=str, default=None,
                        help="Hugging Face model URL to log in W&B run summary.")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
