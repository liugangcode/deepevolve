# molecule

## Problem Description

This task focuses on general molecular property prediction, utilizing the Side Effect Resource (SIDER) dataset as a proxy for algorithm development. The primary goal is to design algorithms that generalize across various molecular property prediction tasks. The dataset is scaffold-split to assess the generalization capabilities of the algorithms when encountering novel chemical structures.

- **Evaluation Metric**: auc
- **Interface File**: `deepevolve_interface.py`

## Initial Idea

The initial approach adopts the concept of "Graph Rationalization with Environment-based Augmentations". This method is designed to enhance interpretability and improve generalization through the use of environment-based augmentations within graph neural network frameworks.

For further insights into the approach, please refer to the following resources:

- [Graph Rationalization with Environment-based Augmentations (Research Paper)](https://arxiv.org/abs/2206.02886)
- [Supplementary Materials on GitHub](https://github.com/liugangcode/GREA)

---

This README provides a concise overview of the problem and the initial idea, outlining the main objectives, evaluation metrics, and key references for further exploration.