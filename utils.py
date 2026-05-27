"""
utils.py — Shared helpers for the Goodreads Genre Classification pipeline.

Contains:
  - Global configuration constants
  - Genre URL dictionary
  - MyDataset  : PyTorch Dataset wrapper for HuggingFace encodings
  - build_label_maps : builds label2id / id2label from a list of string labels
  - compute_metrics  : accuracy + weighted-F1 for HuggingFace Trainer
"""

import os
import torch
from sklearn.metrics import accuracy_score, f1_score

# ── Configuration constants ────────────────────────────────────────────────────

MODEL_NAME  = "distilbert-base-cased"   # HuggingFace model identifier
MAX_LENGTH  = 512                        # Maximum token length for tokenizer
MODEL_SAVE_DIR = "distilbert-reviews-genres"  # Where the fine-tuned model is saved
DATA_DIR    = "data"                     # Directory for preprocessed data artefacts

# Prefer CUDA GPU if available, otherwise CPU
DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

# ── Genre → URL mapping ────────────────────────────────────────────────────────

GENRE_URLS = {
    "poetry":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children":               "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic":         "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":            "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


# ── Dataset class ──────────────────────────────────────────────────────────────

class MyDataset(torch.utils.data.Dataset):
    """
    Wraps a HuggingFace BatchEncoding and a list of integer labels into a
    PyTorch Dataset that the HuggingFace Trainer can consume directly.

    Args:
        encodings : BatchEncoding returned by a HuggingFace tokenizer.
        labels    : List of integer label ids (one per example).
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ── Label map builder ──────────────────────────────────────────────────────────

def build_label_maps(labels):
    """
    Build deterministic label ↔ integer mappings from a flat list of string labels.

    Sorting guarantees the same mapping across separate script runs, which is
    critical when the model is saved and reloaded later.

    Args:
        labels : Iterable of string labels (may contain duplicates).

    Returns:
        label2id : dict  {label_string -> int}
        id2label : dict  {int -> label_string}
    """
    unique_labels = sorted(set(labels))          # sorted → reproducible ordering
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


# ── Metrics for HuggingFace Trainer ───────────────────────────────────────────

def compute_metrics(pred):
    """
    Compute accuracy and weighted F1 score from a HuggingFace EvalPrediction object.

    This function is passed directly to the Trainer via the `compute_metrics`
    argument.  The Trainer calls it after every evaluation step.

    Args:
        pred : transformers.EvalPrediction
               Has attributes `predictions` (raw logits, shape [N, num_labels])
               and `label_ids` (true integer labels, shape [N]).

    Returns:
        dict with keys "accuracy" and "f1".
    """
    true_labels = pred.label_ids
    pred_labels = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1":       f1_score(true_labels, pred_labels, average="weighted"),
    }
