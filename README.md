# MLOps Assignment 2 — Goodreads Genre Classification

Fine-tuning DistilBERT to classify book reviews by genre using the UCSD Goodreads dataset.

## Project Description
This project fine-tunes a pre-trained DistilBERT model on Goodreads book reviews
to predict the genre of a book from its review text. The pipeline includes
data loading, model fine-tuning on Kaggle GPU, experiment tracking with W&B,
and model publishing on Hugging Face Hub.

## Setup Instructions
```bash
pip install -r requirements.txt
python data.py
python train.py
python eval.py
```

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.5713  (57.1%)  |
| F1 Score  | 0.5677           |
| Eval Loss | 2.4547           |


## Links
- Kaggle Notebook: (https://www.kaggle.com/code/sonukumarg25ait2110/notebooka67b3c844b)
- Hugging Face Model: [https://huggingface.co/Sonu-kumar-IIT/distilbert-goodreads-genres](https://huggingface.co/Sonu-kumar-IIT/distilbert-goodreads-genres)
- W&B Dashboard: (https://wandb.ai/g25ait2110-prom-iit-rajasthan/mlops-assignment2/runs/m69vvmnf?nw=nwuserg25ait2110)
