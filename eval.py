"""
eval.py — Evaluate the fine-tuned model on the held-out test set.

Usage:
    python eval.py
    python eval.py --model_dir distilbert-reviews-genres --data_dir data
    python eval.py --wandb --wandb_project mlops-assignment2 --output_dir eval_results

Prerequisites:
    1. Run data.py  — to generate preprocessed artefacts in `data_dir/`.
    2. Run train.py — to fine-tune and save the model to `model_dir/`.

What this script does:
  1. Loads the fine-tuned model and tokenizer from `model_dir/`.
  2. Loads test encodings and labels from `data_dir/`.
  3. Runs trainer.evaluate() to get summary metrics (accuracy, F1, loss).
  4. Runs trainer.predict() to obtain per-example predictions.
  5. Prints a full sklearn classification_report.
  6. Saves the classification report as JSON to `output_dir/eval_report.json`.
  7. Optionally logs final metrics and uploads the report as a W&B Artifact.
"""

import argparse
import json
import os
import pickle
import sys

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import classification_report

from utils import (
    MODEL_SAVE_DIR,
    DATA_DIR,
    DEVICE_NAME,
    MyDataset,
    compute_metrics,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_path(path: str, label: str):
    """Exit with an informative message if a required path is missing."""
    if not os.path.exists(path):
        print(
            f"[eval.py] ERROR: {label} not found: {path}\n"
            "          Make sure data.py and train.py have been run first.",
            file=sys.stderr,
        )
        sys.exit(1)


def load_pickle(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_label_maps(data_dir: str):
    path = os.path.join(data_dir, "label_maps.json")
    check_path(path, "label_maps.json")
    with open(path, "r", encoding="utf-8") as fh:
        maps = json.load(fh)
    label2id = maps["label2id"]
    id2label = {int(k): v for k, v in maps["id2label"].items()}
    return label2id, id2label


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    print(f"\n[eval.py] Using device: {DEVICE_NAME}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load label maps ─────────────────────────────────────────────────────
    label2id, id2label = load_label_maps(args.data_dir)
    class_names = [id2label[i] for i in range(len(id2label))]
    print(f"[eval.py] {len(class_names)} classes: {class_names}")

    # ── 2. Load test artefacts ─────────────────────────────────────────────────
    test_enc_path    = os.path.join(args.data_dir, "test_encodings.pkl")
    test_labels_path = os.path.join(args.data_dir, "test_labels_encoded.pkl")
    test_raw_path    = os.path.join(args.data_dir, "test_labels_raw.pkl")

    check_path(test_enc_path,    "test_encodings.pkl")
    check_path(test_labels_path, "test_labels_encoded.pkl")

    test_encodings      = load_pickle(test_enc_path)
    test_labels_encoded = load_pickle(test_labels_path)
    test_labels_raw     = load_pickle(test_raw_path) if os.path.exists(test_raw_path) else None

    test_dataset = MyDataset(test_encodings, test_labels_encoded)
    print(f"[eval.py] Test set: {len(test_dataset)} examples")

    # ── 3. Load fine-tuned model and tokenizer ────────────────────────────────
    check_path(args.model_dir, "model directory")
    print(f"\n[eval.py] Loading model from '{args.model_dir}/' ...")
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)
    model.to(DEVICE_NAME)
    model.eval()

    # ── 4. Minimal TrainingArguments (evaluation only) ────────────────────────
    # W&B is configured later; disable it here to avoid duplicate init.
    os.environ.setdefault("WANDB_DISABLED", "true")
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.eval_batch_size,
        report_to="none",
        fp16=args.fp16 and DEVICE_NAME == "cuda",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # ── 5. Summary metrics ─────────────────────────────────────────────────────
    print("\n[eval.py] Running evaluation ...")
    eval_results = trainer.evaluate()
    print("\n── Evaluation results ──────────────────────────────────────────────")
    for k, v in eval_results.items():
        print(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")

    # ── 6. Per-example predictions ────────────────────────────────────────────
    print("\n[eval.py] Computing predictions ...")
    pred_output     = trainer.predict(test_dataset)
    pred_ids        = pred_output.predictions.argmax(-1).tolist()
    pred_labels_str = [id2label[i] for i in pred_ids]

    # True labels as strings
    if test_labels_raw:
        true_labels_str = test_labels_raw
    else:
        true_labels_str = [id2label[i] for i in test_labels_encoded]

    # ── 7. Classification report ──────────────────────────────────────────────
    report_str  = classification_report(true_labels_str, pred_labels_str)
    report_dict = classification_report(
        true_labels_str, pred_labels_str, output_dict=True
    )

    print("\n── Classification Report ───────────────────────────────────────────")
    print(report_str)

    # ── 8. Save report as JSON ────────────────────────────────────────────────
    report_path = os.path.join(args.output_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2)
    print(f"[eval.py] Classification report saved to: {report_path}")

    # ── 9. Optional W&B logging ───────────────────────────────────────────────
    if args.wandb:
        try:
            import wandb
            # Unset the env-var override so wandb can initialise normally
            os.environ.pop("WANDB_DISABLED", None)

            wandb.init(
                project=args.wandb_project,
                name=f"{args.wandb_run}-eval",
            )

            # Log scalar metrics
            wandb.log({
                "final/loss":     eval_results.get("eval_loss", 0),
                "final/accuracy": eval_results.get("eval_accuracy", 0),
                "final/f1":       eval_results.get("eval_f1", 0),
            })

            # Upload classification report as a versioned Artifact
            artifact = wandb.Artifact("eval-report", type="evaluation")
            artifact.add_file(report_path)
            wandb.log_artifact(artifact)

            print(f"[eval.py] Metrics and report artifact uploaded to W&B "
                  f"(project: {args.wandb_project}).")
            wandb.finish()

        except ImportError:
            print("[eval.py] [WARNING] wandb not installed. Skipping W&B upload.",
                  file=sys.stderr)
        except Exception as exc:
            print(f"[eval.py] [WARNING] W&B logging failed: {exc}", file=sys.stderr)

    print("\n[eval.py] Evaluation complete.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned DistilBERT model on the held-out test set."
    )

    # Paths
    parser.add_argument("--model_dir",       type=str, default=MODEL_SAVE_DIR,
                        help=f"Directory containing the saved fine-tuned model "
                             f"(default: {MODEL_SAVE_DIR}).")
    parser.add_argument("--data_dir",        type=str, default=DATA_DIR,
                        help=f"Directory containing artefacts from data.py (default: {DATA_DIR}).")
    parser.add_argument("--output_dir",      type=str, default="eval_results",
                        help="Directory to write eval_report.json (default: eval_results).")

    # Eval settings
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Per-device evaluation batch size (default: 16).")
    parser.add_argument("--fp16",            action="store_true",
                        help="Enable FP16 inference (GPU only).")

    # W&B
    parser.add_argument("--wandb",           action="store_true",
                        help="Log final metrics and upload report as a W&B Artifact.")
    parser.add_argument("--wandb_project",   type=str, default="mlops-assignment2",
                        help="W&B project name (default: mlops-assignment2).")
    parser.add_argument("--wandb_run",       type=str, default="distilbert-run-1",
                        help="Base W&B run name; '-eval' is appended (default: distilbert-run-1).")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
