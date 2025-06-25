# Molecule

## Problem Description

This task focuses on the general molecular property prediction, utilizing the Side Effect Resource (SIDER) as a proxy dataset for algorithm development. The primary goal is to design algorithms that can generalize across various molecular property prediction tasks. The dataset is scaffold-split to assess the generalization to novel chemical structures.

- **Evaluation Metric**: AUC (Area Under the Curve)
- **Interface File**: `deepevolve_interface.py`

## Initial Idea

The initial approach for tackling this problem is based on the concept of **Graph Rationalization with Environment-based Augmentations**. This method is detailed in the paper titled "Graph Rationalization with Environment-based Augmentations", which can be accessed [here](https://arxiv.org/abs/2206.02886). The approach focuses on enhancing the generalization capabilities of algorithms by incorporating environment-based augmentations into the graph rationalization process.

For further technical details and implementation, please refer to the supplementary resources available on [GitHub](https://github.com/liugangcode/GREA). This repository contains code and additional documentation to support the understanding and application of the proposed method.