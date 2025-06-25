# usp_p2p

## Problem Description

In this competition, you will build a model that extracts relevant information by matching key phrases in patent documents. The main goal is to determine the semantic similarity between phrases, which is critically important during the patent search and examination process. For example, if one invention claims "television set" while a prior publication describes "TV set", an ideal model would recognize these as semantically equivalent and help patent attorneys or examiners retrieve the relevant documents. This challenge extends beyond simple paraphrase identification. In certain cases, matching can involve recognizing that different terms such as "strong material" and "steel" could refer to equivalent concepts within specific domains, even though their literal meanings might differ.

Key points of note:
- The similarity between two phrases is scored on a scale from 0 (not at all similar) to 1 (identical in meaning).
- Unlike typical semantic similarity tasks, the similarity here is scored within the context provided by the Cooperative Patent Classification (CPC) system (version 2021.05). The CPC code indicates the technical domain of the patent and is used as an additional feature to disambiguate situations where general language similarities might be misleading.
- Models are evaluated based on the Pearson correlation coefficient between the predicted and the actual similarity scores.

Data Files:
- **train.csv**: Contains the training set with phrases, contexts, and their similarity scores.
- **test.csv**: The test set, identical in structure to the training set, but includes true scores.

Data Columns:
- **id**: Unique identifier for a phrase pair
- **anchor**: The first phrase
- **target**: The second phrase
- **context**: The CPC classification indicating the subject area (version 2021.05)
- **score**: The similarity score, sourced from manual expert ratings

Score Meanings:
- 1.0: Very close match (usually exact match except minor differences in conjugation, quantity, or stopwords)
- 0.75: Close synonym or abbreviation (e.g., "mobile phone" vs. "cellphone", "TCP" → "transmission control protocol")
- 0.5: Synonyms with different breadth (hyponym/hypernym matches)
- 0.25: Somewhat related (same high-level domain or antonyms)
- 0.0: Unrelated

Additional Details:
- **Evaluation Metric**: [pearson_correlation]
- **Interface File**: `deepevolve_interface.py`

You can also refer to the USPTO website for more information on CPC codes and check the CPC archive website for details on version 2021.05.

## Initial Idea

The initial approach is to fine-tune the Patent BERT model on the USP-P2P dataset. Specifically, the procedure is as follows:

1. Use the pre-trained model [`anferico/bert-for-patents`](https://huggingface.co/anferico/bert-for-patents) as the base.
2. Modify the model by adding a single-label regression head to handle the regression task required by the similarity score.
3. Preprocess each example by tokenizing the input as a single sequence. This is achieved by joining the anchor, target, and context with the `[SEP]` token.
4. Fine-tune the model for one epoch using:
   - Batch size: 160
   - Learning rate: 2e-5
5. The training process is executed without checkpointing or logging.
6. Finally, evaluate the fine-tuned model on the test set by computing the Pearson correlation coefficient between the predicted and actual scores.

The approach leverages domain-specific pre-training and efficient tokenization to ensure that the context provided by CPC codes is effectively incorporated into the model's predictions.

For mathematical details, if you need to render any equations using MathJax, you can use the following syntax:
$$
\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$
where \( y_i \) is the true similarity score and \( \hat{y}_i \) is the predicted score.

This project aims to efficiently connect the dots between millions of patent documents by providing robust semantic similarity estimations within a specified technical context.