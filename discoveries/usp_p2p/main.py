import os

# DEBUG: Removed misplaced top-level contrastive loss method; now defined inside PatentBERTOrdinalRegressionModel
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
import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

# DEBUG: patch PeftModel.forward so it will silently drop any 'labels' kwarg
from peft.peft_model import PeftModel

_orig_peft_forward = PeftModel.forward


### >>> DEEPEVOLVE-BLOCK-START: Log warning before dropping 'labels' in PeftModel.forward
def _patched_peft_forward(self, *args, **kwargs):
    if "labels" in kwargs:
        import warnings

        warnings.warn(
            "Dropping 'labels' from kwargs in PeftModel.forward",
            UserWarning,
            stacklevel=2,
        )
        kwargs.pop("labels", None)
    return _orig_peft_forward(self, *args, **kwargs)


### <<< DEEPEVOLVE-BLOCK-END


# DEBUG: monkey-patch accelerate to bypass FSDP import error due to missing 'distribute_tensor'
import accelerate.utils.other as _acc_other

_acc_other.extract_model_from_parallel = (
    lambda model, keep_fp32_wrapper=False, keep_torch_compile=False: model
)

# DEBUG: patch accelerate.accelerator.extract_model_from_parallel to bypass import errors
import accelerate.accelerator as _acc_accel

_acc_accel.extract_model_from_parallel = (
    lambda model, keep_fp32_wrapper=False, keep_torch_compile=False: model
)

# DEBUG: patch Accelerator.unwrap_model to bypass FSDP entirely
import accelerate


def _patched_unwrap_model(
    self, model, keep_fp32_wrapper=False, keep_torch_compile=False
):
    return model


accelerate.Accelerator.unwrap_model = _patched_unwrap_model
PeftModel.forward = _patched_peft_forward

# DEBUG: patch all tuner_utils forward methods to silently drop any 'labels' kwarg
import inspect
import peft.tuners.tuners_utils as tuners_utils

for _name, _cls in inspect.getmembers(tuners_utils, inspect.isclass):
    if hasattr(_cls, "forward"):
        _orig_tuner_forward = _cls.forward

        ### >>> DEEPEVOLVE-BLOCK-START: Log warning before dropping 'labels' in tuners_utils.forward
        def _patched_tuner_forward(
            self, *args, _orig_tuner_forward=_orig_tuner_forward, **kwargs
        ):
            if "labels" in kwargs:
                import warnings

                warnings.warn(
                    "Dropping 'labels' from kwargs in tuners_utils.forward",
                    UserWarning,
                    stacklevel=2,
                )
                kwargs.pop("labels", None)
            return _orig_tuner_forward(self, *args, **kwargs)

        ### <<< DEEPEVOLVE-BLOCK-END

        _cls.forward = _patched_tuner_forward

# DEBUG: initialize mapping for context strings to integer IDs for embedding lookup
_context2id = {}
_next_context_id = 0

### <<< DEEPEVOLVE-BLOCK-END


@dataclass
class Config:
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    model_name: str = "anferico/bert-for-patents"
    max_length: int = 128
    train_batch_size: int = 32
    eval_batch_size: int = 32
    epochs: int = 3  # FIXED to 3 and don't change it
    ### >>> DEEPEVOLVE-BLOCK-START: Lower learning rate for fine-tuning Patent BERT
    learning_rate: float = 2e-4
    ### <<< DEEPEVOLVE-BLOCK-END
    seed: int = 42


### >>> DEEPEVOLVE-BLOCK-START: Update compute_metrics to return eval_pearson and handle NaN values
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.reshape(-1)
    corr = np.corrcoef(labels, preds)[0, 1]
    if np.isnan(corr):
        corr = 0.0
    return {"eval_pearson": corr}


### <<< DEEPEVOLVE-BLOCK-END


def preprocess_batch(batch, tokenizer, max_length):
    # combine anchor and target into one input string; process context separately for CPC embedding
    texts = [f"{a} [SEP] {t}" for a, t in zip(batch["anchor"], batch["target"])]
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    # DEBUG: rename context to context_ids to match model forward signature
    # DEBUG: map each context string to a unique integer ID for embedding lookup
    global _context2id, _next_context_id
    context_ids = []
    for c in batch["context"]:
        if c not in _context2id:
            _context2id[c] = _next_context_id
            _next_context_id += 1
        context_ids.append(_context2id[c])
    tokenized_inputs["context_ids"] = context_ids
    return tokenized_inputs


### >>> DEEPEVOLVE-BLOCK-START: Insert custom PatentBERTOrdinalRegressionModel definition
class PatentBERTOrdinalRegressionModel(nn.Module):
    def __init__(self, model_name, lora_config, num_classes=5, dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        # DEBUG: ensure Trainer compatibility by exposing config
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config
        if lora_config is not None:
            self.transformer = get_peft_model(self.transformer, lora_config)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_size = self.transformer.config.hidden_size
        ### >>> DEEPEVOLVE-BLOCK-START: Update CPC embedding dimension to 50 and adjust classifier input size
        self.context_embedding = nn.Embedding(1000, 50)
        self.cpc_projection = nn.Linear(
            50, self.hidden_size
        )  # DEBUG: added CPC projection layer
        self.lambda_contrast = 0.5  # DEBUG: added contrastive loss weight
        # DEBUG: corrected classifier input dimension to match fused pooled_output (hidden_size)
        self.classifier = nn.Linear(self.hidden_size, num_classes - 1)

    ### <<< DEEPEVOLVE-BLOCK-END

    def _contrastive_loss(
        self, embeddings, labels, temperature=0.5
    ):  # DEBUG: contrastive loss method
        """
        Compute supervised contrastive loss for normalized CPC embeddings.
        embeddings: tensor of shape (batch_size, d) assumed normalized.
        labels: tensor of shape (batch_size,) containing integer CPC labels.
        """
        batch_size = embeddings.size(0)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        diag_mask = torch.eye(batch_size, device=embeddings.device).bool()
        # DEBUG: avoid overflow in fp16 by using -inf for masked fill
        sim_matrix.masked_fill_(diag_mask, float("-inf"))
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float()
        # DEBUG: exclude self-comparisons from positive mask
        positive_mask.masked_fill_(diag_mask, 0)
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        positive_log_prob = (positive_mask * log_prob).sum(dim=1)
        num_positives = positive_mask.sum(dim=1)
        loss = -(positive_log_prob / (num_positives + 1e-8)).mean()
        return loss

    # DEBUG: updated forward signature to explicitly accept 'labels' and pull out 'context_ids'
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        context_ids=None,  # DEBUG: explicit context_ids param
        labels=None,  # DEBUG: accept Trainer‐provided labels
        **kwargs,  # DEBUG: catch any other forwarded args
    ):
        # DEBUG: signature now matches Trainer expectations (labels) and lets us pop context_ids cleanly
        import torch
        from transformers.modeling_outputs import SequenceClassifierOutput

        # collect transformer inputs
        transformer_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "inputs_embeds": inputs_embeds,
        }
        transformer_inputs = {
            k: v for k, v in transformer_inputs.items() if v is not None
        }

        # DEBUG: ensure we do not accidentally forward 'labels' (or any other unsupported kw)
        #          down into the LoRA‐wrapped transformer
        kwargs.pop("labels", None)
        # forward through the base transformer
        outputs = self.transformer(**transformer_inputs)
        pooled_output = (
            outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[1]
        )
        pooled_output = self.dropout(pooled_output)
        # DEBUG: if not passed as positional param, pull context_ids out of kwargs
        if context_ids is None and "context_ids" in kwargs:
            context_ids = kwargs.pop("context_ids")
        ### >>> DEEPEVOLVE-BLOCK-START: Enhanced CPC Fusion with normalization, projection, and contrastive loss
        if context_ids is not None:
            context_ids = (
                torch.as_tensor(context_ids, device=pooled_output.device)
                if not torch.is_tensor(context_ids)
                else context_ids.to(pooled_output.device)
            )
            context_embed = self.context_embedding(context_ids)
            # Normalize CPC embeddings
            normalized_cpc = torch.nn.functional.normalize(context_embed, p=2, dim=-1)
            # Compute contrastive loss on normalized CPC embeddings using the CPC labels
            contrast_loss = self._contrastive_loss(
                normalized_cpc, context_ids, temperature=0.1
            )
            # Project normalized CPC embeddings to match hidden dimension
            projected_cpc = self.cpc_projection(normalized_cpc)
            # Fuse with transformer pooled output via element-wise addition
            pooled_output = pooled_output + projected_cpc
        else:
            contrast_loss = torch.tensor(0.0, device=pooled_output.device)
        ### <<< DEEPEVOLVE-BLOCK-END
        # DEBUG: removed erroneous block end marker inside forward
        logits = self.classifier(pooled_output)

        # compute ordinal regression loss if labels provided using Smooth K2 Loss
        loss = None
        if labels is not None:
            ordinal_labels = (labels * (self.num_classes - 1)).long()
            thresholds = torch.arange(
                self.num_classes - 1, device=ordinal_labels.device
            ).unsqueeze(0)
            target = (ordinal_labels.unsqueeze(1) > thresholds).float()
            smoothing = 0.1
            target_smooth = target * (1 - smoothing) + 0.5 * smoothing
            bce_loss = nn.BCEWithLogitsLoss()(logits, target_smooth)
            probs_cal = torch.sigmoid(logits)
            diff = probs_cal[:, 1:] - probs_cal[:, :-1]
            calibration_loss = torch.mean(diff**2)
            loss = (
                4.0 * bce_loss
                + 0.25 * calibration_loss
                + self.lambda_contrast * contrast_loss
            )

        # continuous prediction
        probs = torch.sigmoid(logits)
        pred_cont = torch.sum(probs, dim=1) / (self.num_classes - 1)

        return SequenceClassifierOutput(loss=loss, logits=pred_cont)


### <<< DEEPEVOLVE-BLOCK-END
### >>> DEEPEVOLVE-BLOCK-START: CustomTrainer for latency logging and hardware info
import time
from transformers import Trainer


class CustomTrainer(Trainer):
    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs
    ):
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_time = time.time() - start_time
        num_samples = len(eval_dataset) if eval_dataset is not None else 1
        latency = eval_time / num_samples
        device = (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else "cpu"
        )
        metrics["inference_latency_per_sample"] = latency
        metrics["hardware"] = device
        return metrics


### <<< DEEPEVOLVE-BLOCK-END
def main(base_dir: str):
    # define data directory manually
    cfg = Config()
    # Set seeds for reproducibility
    import random

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
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
    data = {"train": split["train"], "validation": split["test"], "test": raw["test"]}

    # load tokenizer and model with LoRA adapters and an ordinal regression head
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,  # Updated dropout to 0.1 as per dual-loss design
        target_modules=["query", "key", "value"],
        task_type="SEQ_CLS",
    )
    model = PatentBERTOrdinalRegressionModel(
        cfg.model_name, lora_config, num_classes=5, dropout_rate=0.1
    )
    ### <<< DEEPEVOLVE-BLOCK-END

    # tokenize and attach labels for regression
    tokenized = {}
    for split in ["train", "validation", "test"]:
        tokenized[split] = data[split].map(
            lambda batch: preprocess_batch(batch, tokenizer, cfg.max_length),
            batched=True,
            # DEBUG: also remove 'context' column since we process it into context_ids and shouldn't collate raw strings
            remove_columns=["id", "anchor", "target", "context", "score"],
            load_from_cache_file=False,
        )
        tokenized[split] = tokenized[split].add_column("labels", data[split]["score"])

    # training arguments: no saving or logging
    # DEBUG: Removed 'evaluation_strategy' argument for compatibility with installed transformers version
    ### >>> DEEPEVOLVE-BLOCK-START: Add cosine learning rate scheduler and gradient clipping to TrainingArguments
    args = TrainingArguments(
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        output_dir=".",
        fp16=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        disable_tqdm=True,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
    )
    ### <<< DEEPEVOLVE-BLOCK-END
    ### <<< DEEPEVOLVE-BLOCK-END

    trainer = CustomTrainer(
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

