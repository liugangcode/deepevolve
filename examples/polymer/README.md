# Polymer

## Problem Description

### Overview

Can your model unlock the secrets of polymers? In this competition, you're tasked with predicting the fundamental properties of polymers to speed up the development of new materials. Your contributions will help researchers innovate faster, paving the way for more sustainable and biocompatible materials that can positively impact our planet.

### Description

Polymers are the essential building blocks of our world, from the DNA within our bodies to the plastics we use every day. They are key to innovation in critical fields like medicine, electronics, and sustainability. The search for the next generation of groundbreaking, eco-friendly materials is on, and machine learning can be the solution. However, progress has been stalled by one major hurdle: a critical lack of accessible, high-quality data.

Our Open Polymer Prediction 2025 introduces a game-changing, large-scale open-source dataset - ten times larger than any existing resource. We invite you to piece together the missing links and unlock the vast potential of sustainable materials.

Your mission is to predict a polymer's real-world performance directly from its chemical structure. You'll be provided with a polymer's structure as a simple text string (SMILES), and your challenge is to build a model that can accurately forecast five key metrics that determine how it will behave. This includes predicting its density, its response to heat (thermal conductivity, Tc) and glass transition temperature (Tg), and its fundamental molecular size and packing efficiency (radius of gyration, Rg, and fractional free volume, FFV). The ground truth for this competition is averaged from multiple runs of molecular dynamics simulation.

Your contributions have the potential to redefine polymer discovery, accelerating sustainable polymer research through virtual screening and driving significant advancements in materials science.

### Evaluation Metric

The evaluation metric for this contest is a weighted Mean Absolute Error (wMAE) across five polymer properties, defined as:

$$
\mathrm{wMAE} = \frac{1}{\lvert \mathcal{X} \rvert} \sum_{X \in \mathcal{X}} \sum_{i \in I(X)} w_{i}\, \bigl\lvert \hat{y}_{i}(X) \;- y_{i}(X)\bigr\rvert
$$

To ensure that all property types contribute equally regardless of their scale or frequency, each property is given a weight \(w_{i}\):

$$
w_{i} = \frac{1}{r_{i}} \;\times\; \frac{K\,\sqrt{\tfrac{1}{n_{i}}}}{\displaystyle\sum_{j=1}^{K}\sqrt{\tfrac{1}{n_{j}}}}
$$

- **Evaluation Metric**: Combination of weighted MAE and R^2
- **Interface File**: `deepevolve_interface.py`

### Task

In this competition, your task is to use polymer structure data (SMILES) to predict five key chemical properties derived from molecular-dynamics simulation:

- **Glass transition temperature** (`Tg`)
- **Fractional free volume** (`FFV`)
- **Thermal conductivity** (`Tc`)
- **Polymer density** (`Density`)
- **Radius of gyration** (`Rg`)

Successfully predicting these properties is crucial for scientists to accelerate the design of novel polymers with targeted characteristics, which can be used in various applications.

### Data Files

#### `train/valid/test.csv`

| Column    | Description                                              |
|-----------|----------------------------------------------------------|
| `id`      | Unique identifier for each polymer.                      |
| `SMILES`  | Sequence-like chemical notation of polymer structures.   |
| `Tg`      | Glass transition temperature (°C).                       |
| `FFV`     | Fractional free volume.                                  |
| `Tc`      | Thermal conductivity (W/m·K).                            |
| `Density` | Polymer density (g/cm³).                                 |
| `Rg`      | Radius of gyration (Å).                                  |

## Initial Idea

### Graph Rationalization with Environment-based Augmentations

The initial approach involves using graph rationalization techniques with environment-based augmentations to improve the model's ability to predict polymer properties. This method leverages the structural information encoded in the SMILES representation of polymers, enhancing the model's interpretability and performance.

For more details, you can refer to the paper [Graph Rationalization with Environment-based Augmentations](https://arxiv.org/abs/2206.02886) and the supplementary resources available on [GitHub](https://github.com/liugangcode/GREA).