# USP-P2P Semantic Similarity Challenge

## Overview

In this competition, participants are tasked with building a model to determine the semantic similarity between pairs of phrases extracted from patent documents. The goal is to assist patent attorneys and examiners in identifying whether an invention has been described before. This is achieved by matching key phrases and their contexts within patent documents using the Cooperative Patent Classification (CPC) system.

## Problem Description

Patent documents contain rich technical content and the phrasing used can vary significantly. For example, a model should be able to recognize that the phrases "television set" and "TV set" refer to the same device. Moreover, the model must account for context provided by CPC codes (version 2021.05) which indicate the technical domain. Thus, the task extends beyond simple paraphrase identification to include cases such as matching "strong material" with "steel", where the interpretation can vary by domain. 

### Technical Challenge

Given pairs of phrases (an anchor and a target), alongside a contextual feature defined by the CPC code, your model must predict a similarity score between 0 and 1:
- **0.0**: Unrelated
- **0.25**: Somewhat related (e.g., same high-level domain or even antonyms)
- **0.5**: Synonyms with different breadth (hyponym/hypernym matches)
- **0.75**: Close synonym or abbreviation (e.g., "mobile phone" vs. "cellphone", "TCP" vs. "transmission control protocol")
- **1.0**: Very close match (usually an almost exact match, barring minor differences)

The model’s performance is evaluated using the Pearson correlation coefficient between the predicted and actual similarity scores.

## Data Description

The dataset provided for this challenge consists of the following files:

- **train.csv**: The training set containing the phrases, contextual CPC classification, and their similarity scores.
- **test.csv**: The test set, which mirrors the structure of the training set and includes true scores for evaluation.

### Data Columns

Each entry in the dataset consists of:
- **id**: Unique identifier for a phrase pair.
- **anchor**: The first phrase.
- **target**: The second phrase.
- **context**: The CPC classification (version 2021.05) indicating the subject area within which similarity is scored.
- **score**: The similarity score (floating point number between 0 and 1) obtained from manual expert ratings.

## Evaluation Metric

The submission will be evaluated based on the Pearson correlation coefficient between the predicted similarity scores and the actual scores. Mathematically, this is represented as:

```math
r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}
```

where $x_i$ and $y_i$ are the predicted and actual scores respectively, and $\bar{x}$ and $\bar{y}$ are their means.

## Baseline Approach

### Fine-tuning the Patent BERT Model

An initial approach to address this challenge is to fine-tune the [BERT for Patents](https://huggingface.co/anferico/bert-for-patents) model. The following steps outline the proposed methodology:

1. **Model Selection**: Start with the `anferico/bert-for-patents` model.
2. **Architecture Modification**: Attach a single-label regression head on the model to predict the similarity score.
3. **Data Tokenization**: Tokenize each example by concatenating the anchor phrase, target phrase, and context with a `[SEP]` token. This results in an input format similar to:
   ```
   anchor [SEP] target [SEP] context
   ```
4. **Training**: 
   - Fine-tune for one epoch.
   - Use a batch size of 160.
   - Set a learning rate of $2 \times 10^{-5}$.
   - Training is conducted without checkpointing or logging.
5. **Evaluation**: Evaluate the fine-tuned model on the test set by computing the Pearson correlation between the predictions and the provided similarity scores.

## Competition Details

- **Interface**: The competition code should implement the interface defined in `deepevolve_interface.py`.
- **Test Set**: The unseen test set contains approximately 12,000 phrase pairs. Note that a small public test set has been provided for preliminary testing, but it is not used for scoring.

## Resources

- **Patent BERT Model**: [anferico/bert-for-patents](https://huggingface.co/anferico/bert-for-patents)
- **CPC Codes Information**: Detailed information on CPC codes can be found on the USPTO website and the [CPC archive website](https://www.cooperativepatentclassification.org/).