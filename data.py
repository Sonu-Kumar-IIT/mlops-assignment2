"""
data.py — Data loading, sampling, train/test split, and tokenizer encoding.

Usage:
    python data.py
    python data.py --sample_size 500 --train_ratio 0.8 --head 5000
    python data.py --genres poetry romance fantasy_paranormal
    python data.py --data_dir my_data --cache_reviews

Steps performed:
  1. Download a gzip-compressed JSON review file for each genre via HTTP streaming.
  2. Sample up to `sample_size` reviews per genre.
  3. Split each genre into train (first `train_ratio` fraction) and test (remainder).
  4. Build label2id / id2label maps and save them as JSON.
  5. Tokenize all texts with the DistilBERT tokenizer.
  6. Pickle train/test encodings and encoded integer labels to `data_dir/`.

Downstream scripts (train.py, eval.py) load the artefacts from `data_dir/`.
"""

import argparse
import gzip
import json
import os
import pickle
import random
import sys

import requests
from transformers import DistilBertTokenizerFast

from utils import (
    GENRE_URLS,
    MODEL_NAME,
    MAX_LENGTH,
    DATA_DIR,
    build_label_maps,
)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_reviews(url: str, head: int = 10_000, sample_size: int = 2_000) -> list:
    """
    Stream a gzip-compressed JSON review file from `url`, collect up to `head`
    reviews, then return a random sample of `sample_size` reviews.

    Args:
        url         : Direct HTTP URL to a .json.gz file.
        head        : Maximum number of reviews to read from the stream.
        sample_size : Number of reviews to randomly sample from `head` reviews.

    Returns:
        List of review text strings (length <= sample_size).
    """
    reviews = []
    print(f"  Streaming {url} ...")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with gzip.open(response.raw, "rt", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                text   = record.get("review_text", "").strip()
                if text:
                    reviews.append(text)
                if head is not None and len(reviews) >= head:
                    break
    except Exception as exc:
        print(f"  [WARNING] Could not load {url}: {exc}", file=sys.stderr)
        return []

    sampled = random.sample(reviews, min(sample_size, len(reviews)))
    print(f"  Loaded {len(reviews)} reviews, sampled {len(sampled)}.")
    return sampled


# ── Train / test split ─────────────────────────────────────────────────────────

def split_genre_reviews(genre_reviews_dict: dict, train_ratio: float = 0.8):
    """
    Split each genre's review list into train and test sets.

    For reproducibility a fixed `random.seed` should be set before calling
    this function.

    Args:
        genre_reviews_dict : {genre_string -> [review_text, ...]}
        train_ratio        : Fraction of each genre to use for training.

    Returns:
        train_texts, train_labels, test_texts, test_labels  (all flat lists)
    """
    train_texts, train_labels = [], []
    test_texts,  test_labels  = [], []

    for genre, reviews in genre_reviews_dict.items():
        n_train = int(len(reviews) * train_ratio)
        for text in reviews[:n_train]:
            train_texts.append(text)
            train_labels.append(genre)
        for text in reviews[n_train:]:
            test_texts.append(text)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)

    # ── 1. Determine genres ────────────────────────────────────────────────────
    genres_to_load = args.genres if args.genres else list(GENRE_URLS.keys())
    print(f"\n[data.py] Genres selected: {genres_to_load}")

    # ── 2. Load (or restore cached) reviews ───────────────────────────────────
    cache_path = os.path.join(args.data_dir, "genre_reviews_dict.pickle")

    if args.cache_reviews and os.path.exists(cache_path):
        print(f"[data.py] Loading cached reviews from {cache_path}")
        with open(cache_path, "rb") as fh:
            genre_reviews_dict = pickle.load(fh)
        # Only keep requested genres
        genre_reviews_dict = {g: v for g, v in genre_reviews_dict.items() if g in genres_to_load}
    else:
        genre_reviews_dict = {}
        for genre in genres_to_load:
            if genre not in GENRE_URLS:
                print(f"[data.py] [WARNING] Unknown genre '{genre}', skipping.", file=sys.stderr)
                continue
            print(f"[data.py] Loading genre: {genre}")
            genre_reviews_dict[genre] = load_reviews(
                GENRE_URLS[genre],
                head=args.head,
                sample_size=args.sample_size,
            )
        if args.cache_reviews:
            with open(cache_path, "wb") as fh:
                pickle.dump(genre_reviews_dict, fh)
            print(f"[data.py] Cached reviews saved to {cache_path}")

    if not genre_reviews_dict:
        print("[data.py] ERROR: No reviews loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ── 3. Train / test split ──────────────────────────────────────────────────
    print(f"\n[data.py] Splitting data (train ratio = {args.train_ratio}) ...")
    train_texts, train_labels, test_texts, test_labels = split_genre_reviews(
        genre_reviews_dict, train_ratio=args.train_ratio
    )
    print(f"  Train: {len(train_texts)} examples | Test: {len(test_texts)} examples")

    # ── 4. Build and save label maps ───────────────────────────────────────────
    label2id, id2label = build_label_maps(train_labels)
    label_maps = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    label_maps_path = os.path.join(args.data_dir, "label_maps.json")
    with open(label_maps_path, "w", encoding="utf-8") as fh:
        json.dump(label_maps, fh, indent=2)
    print(f"[data.py] Label maps saved to {label_maps_path}")
    print(f"  Classes ({len(label2id)}): {list(label2id.keys())}")

    # ── 5. Tokenize ────────────────────────────────────────────────────────────
    print(f"\n[data.py] Tokenizing with '{args.model_name}' (max_length={args.max_length}) ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )

    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded  = [label2id[y] for y in test_labels]

    print("  Tokenization complete.")

    # ── 6. Save artefacts ──────────────────────────────────────────────────────
    artefacts = {
        "train_encodings.pkl":       train_encodings,
        "test_encodings.pkl":        test_encodings,
        "train_labels_encoded.pkl":  train_labels_encoded,
        "test_labels_encoded.pkl":   test_labels_encoded,
        "train_labels_raw.pkl":      train_labels,
        "test_labels_raw.pkl":       test_labels,
    }
    for fname, obj in artefacts.items():
        path = os.path.join(args.data_dir, fname)
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)

    print(f"\n[data.py] All artefacts saved to '{args.data_dir}/':")
    for fname in artefacts:
        print(f"  {fname}")
    print("\n[data.py] Done. Run train.py next.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download, preprocess, and encode Goodreads review data for DistilBERT fine-tuning."
    )
    parser.add_argument(
        "--genres", nargs="+", default=None,
        help="Subset of genre keys to load (default: all 8 genres). "
             "Example: --genres poetry romance fantasy_paranormal",
    )
    parser.add_argument(
        "--sample_size", type=int, default=2_000,
        help="Reviews to randomly sample per genre after streaming (default: 2000).",
    )
    parser.add_argument(
        "--head", type=int, default=10_000,
        help="Maximum reviews to read from each compressed file before sampling (default: 10000). "
             "Lower this (e.g. 500) on CPU to speed things up.",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Fraction of each genre used for training (default: 0.8).",
    )
    parser.add_argument(
        "--model_name", type=str, default=MODEL_NAME,
        help=f"HuggingFace model name for the tokenizer (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--max_length", type=int, default=MAX_LENGTH,
        help=f"Tokenizer max sequence length (default: {MAX_LENGTH}).",
    )
    parser.add_argument(
        "--data_dir", type=str, default=DATA_DIR,
        help=f"Directory to write preprocessed artefacts (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--cache_reviews", action="store_true",
        help="Save/load raw reviews as a pickle file to skip re-downloading.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
