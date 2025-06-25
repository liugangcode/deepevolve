# usp_p2p

## Problem Description

In this competition, participants are tasked with training models on a novel semantic similarity dataset to extract relevant information by matching key phrases in patent documents. The goal is to determine the semantic similarity between phrases, which is critically important during the patent search and examination process to ascertain if an invention has been previously described. For instance, if one invention claims "television set" and a prior publication describes "TV set," a model should ideally recognize these as equivalent, aiding patent attorneys or examiners in retrieving pertinent documents.

This challenge extends beyond simple paraphrase identification. For example, if one invention claims a "strong material" and another uses "steel," these may also be considered a match. The definition of a "strong material" varies by domain, which is why the Cooperative Patent Classification (CPC) is included as a technical domain context to help disambiguate such situations.

Participants are required to build a model that matches phrases to extract contextual information, thereby assisting the patent community in connecting the dots between millions of patent documents. Models are evaluated based on the Pearson correlation coefficient between predicted and actual similarity scores.

In the dataset, participants are presented with pairs of phrases (an anchor and a target phrase) and asked to rate their similarity on a scale from 0 (not at all similar) to 1 (identical in meaning). This task is unique as similarity is scored within the context of a patent's CPC classification (version 2021.05), which indicates the subject matter of the patent. For example, while the phrases "bird" and "Cape Cod" may have low semantic similarity in general language, their meanings are more closely related in the context of "house."

This is a code competition where participants submit code to be run against an unseen test set, which contains approximately 12,000 pairs of phrases. A small public test set is provided for testing purposes but is not used in scoring.

Information on CPC codes can be found on the [USPTO website](https://www.uspto.gov/) and the CPC version 2021.05 is available on the [CPC archive website](https://www.cooperativepatentclassification.org/).

Score meanings:
- **1.0**: Very close match (usually exact match except for minor changes in conjugation, quantity, or stopwords).
- **0.75**: Close synonym or abbreviation (e.g., "mobile phone" vs. "cellphone" or "TCP" → "transmission control protocol").
- **0.5**: Synonyms with different breadth (hyponym/hypernym matches).
- **0.25**: Somewhat related (same high-level domain or antonyms).
- **0.0**: Unrelated.

Files:
- `train.csv`: The training set, containing phrases, contexts, and their similarity scores.
- `test.csv`: The test set, identical in structure to the training set but including true scores.

Columns:
- **id**: Unique identifier for a phrase pair.
- **anchor**: The first phrase.
- **target**: The second phrase.
- **context**: The CPC classification (version 2021.05) indicating the subject within which similarity is scored.
- **score**: The similarity value, sourced from one or more manual expert ratings.

- **Evaluation Metric**: Pearson correlation
- **Interface File**: `deepevolve_interface.py`

## Initial Idea

The initial approach involves fine-tuning the [BERT for Patents](https://huggingface.co/anferico/bert-for-patents) model on the USP-P2P dataset. This approach utilizes the `anferico/bert-for-patents` model with a single-label regression head. Each example is tokenized by joining the anchor, target, and context with `[SEP]`. The model is fine-tuned for one epoch with a batch size of 160 and a learning rate of 2e-5, without checkpointing or logging. Finally, the model is evaluated on the test set by computing the Pearson correlation between the predicted and actual scores.