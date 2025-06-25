import os
# disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

@dataclass
class Config:
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    model_name: str = "anferico/bert-for-patents"
    max_length: int = 128
    train_batch_size: int = 16 * 10
    eval_batch_size: int = 16 * 10
    epochs: int = 3 # FIXED to 3 and don't change it
    learning_rate: float = 2e-5
    seed: int = 42

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.reshape(-1)
    corr = np.corrcoef(labels, preds)[0, 1]
    return {"pearson": corr}

def preprocess_batch(batch, tokenizer, max_length):
    # combine anchor, target, and context into one input string
    texts = [
        f"{a} [SEP] {t} [SEP] {c}"
        for a, t, c in zip(batch["anchor"], batch["target"], batch["context"])
    ]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def main(base_dir: str):
    # define data directory manually
    cfg = Config()
    train_path = os.path.join(base_dir, cfg.train_file)
    test_path = os.path.join(base_dir, cfg.test_file)

    # load datasets (test.csv includes true similarity scores)
    raw = load_dataset(
        "csv",
        data_files={"train": train_path, "test": test_path},
        column_names=["id", "anchor", "target", "context", "score"],
        sep=",",
        skiprows=1,
    )

    # split off 20% of train for validation
    split = raw["train"].train_test_split(test_size=0.2, seed=cfg.seed)
    data = {
        "train": split["train"],
        "validation": split["test"],
        "test": raw["test"]
    }

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=1,
        problem_type="regression",
    )

    # tokenize and attach labels for regression
    tokenized = {}
    for split in ["train", "validation", "test"]:
        tokenized[split] = data[split].map(
            lambda batch: preprocess_batch(batch, tokenizer, cfg.max_length),
            batched=True,
            remove_columns=["id", "anchor", "target", "context", "score"],
            load_from_cache_file=False,
        )
        tokenized[split] = tokenized[split].add_column(
            "labels", data[split]["score"]
        )

    # training arguments: no saving or logging
    args = TrainingArguments(
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        output_dir="."
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])

    if test_metrics.get("eval_pearson") is None:
        raise ValueError("Test set metrics don't have the key 'eval_pearson'")
        
    return test_metrics

if __name__ == "__main__":
    base_dir = "../../../data_cache/usp_p2p"
    test_metrics = main(base_dir)
    print("Test set metrics:", test_metrics)