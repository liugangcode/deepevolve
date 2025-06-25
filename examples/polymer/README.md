# Polymer Competition

Welcome to the Polymer Competition! This challenge invites you to develop a machine learning model that predicts the fundamental properties of polymers directly from their chemical structure. Your work will accelerate the discovery of innovative, sustainable, and biocompatible materials with wide-ranging applications.

---

## Overview

Polymers are the essential building blocks of our world—from the DNA within our bodies to everyday plastics. They drive innovation in medicine, electronics, and sustainability. However, the discovery of new, eco-friendly polymer materials has been slow due to a lack of high-quality, accessible data.

The Open Polymer Prediction 2025 challenge introduces a game-changing, large-scale open-source dataset that is ten times larger than any previous resource. In this competition, your mission is to predict a polymer’s real-world performance based solely on its chemical structure (provided in SMILES format). By accurately forecasting five key properties, your model will help scientists accelerate the design and virtual screening of new polymers.

---

## Problem Description

You are provided with polymer data in CSV files (`train.csv`, `valid.csv`, and `test.csv`). Each file includes the following columns:

| Column    | Description                                               |
|-----------|-----------------------------------------------------------|
| `id`      | A unique identifier for each polymer.                   |
| `SMILES`  | A sequence-like chemical notation representing the polymer structure. |
| `Tg`      | Glass transition temperature (°C).                      |
| `FFV`     | Fractional free volume.                                   |
| `Tc`      | Thermal conductivity (W/m·K).                             |
| `Density` | Polymer density (g/cm³).                                  |
| `Rg`      | Radius of gyration (Å).                                   |

Your task is to accurately predict the following properties from the SMILES representation:

- **Glass transition temperature (Tg)**
- **Fractional free volume (FFV)**
- **Thermal conductivity (Tc)**
- **Polymer density (Density)**
- **Radius of gyration (Rg)**

These target variables are averaged from multiple runs of molecular dynamics simulations.

---

## Evaluation Metric

The predictions will be evaluated using a weighted Mean Absolute Error (wMAE) across the five properties. The wMAE is defined as:

```math
\mathrm{wMAE} = \frac{1}{\lvert \mathcal{X} \rvert} \sum_{X \in \mathcal{X}} \sum_{i \in I(X)} w_{i}\,\bigl| \hat{y}_{i}(X) - y_{i}(X)\bigr|
```

Each property is given a weight $w_i$ to ensure equal contribution regardless of scale or frequency. The weight for property $i$ is calculated as:

```math
w_{i} = \frac{1}{r_{i}} \;\times\; \frac{K\,\sqrt{\tfrac{1}{n_{i}}}}{\displaystyle\sum_{j=1}^{K}\sqrt{\tfrac{1}{n_{j}}}}
```

In addition to wMAE, the final evaluation metric is a combination of the weighted MAE and the $R^2$ score.

---

## Task

Your challenge is to build a predictive model that estimates the five key polymer properties from the provided SMILES strings. By doing so, you will play a vital role in enabling rapid, in-silico screening of polymers, ultimately expediting the development of targeted and sustainable materials.

---

## Interface

The competition interface is provided in the file: `deepevolve_interface.py`.

---

## Initial Idea

### Graph Rationalization with Environment-based Augmentations

For an innovative approach to modeling, consider exploring ideas from the paper [Graph Rationalization with Environment-based Augmentations](https://arxiv.org/abs/2206.02886). Additional resources and code are available in the [GREA GitHub repository](https://github.com/liugangcode/GREA).