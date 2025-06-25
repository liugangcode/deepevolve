# Molecule

## Overview
This repository addresses the "molecule" problem, focusing on general molecular property prediction using the Side Effect Resource (SIDER) as a proxy dataset for algorithm development. The primary goal is to design algorithms that exhibit strong generalization across various molecular property prediction tasks. The dataset is scaffold-split to assess the algorithm's ability to generalize to novel chemical structures.

## Problem Description
- **Task**: General molecular property prediction.
- **Dataset**: Side Effect Resource (SIDER), with a scaffold split to evaluate generalization.
- **Evaluation Metric**: Area Under the Curve (AUC), denoted as $auc$.
- **Interface Implementation**: The main interface is implemented in the [`deepevolve_interface.py`](./deepevolve_interface.py) file.

## Initial Idea
### Graph Rationalization with Environment-based Augmentations
- **Reference Paper**: [Graph Rationalization with Environment-based Augmentations](https://arxiv.org/abs/2206.02886)
- **Supplementary Material**: Available at the [GREA GitHub repository](https://github.com/liugangcode/GREA)