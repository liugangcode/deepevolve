# polymer

## Problem Description

Can your model unlock the secrets of polymers? In this competition, you are tasked with predicting the fundamental properties of polymers to speed up the development of new materials. Your contributions will help researchers innovate faster, paving the way for more sustainable and biocompatible materials that can positively impact our planet.

Polymers are the essential building blocks of our world, from the DNA within our bodies to the plastics we use every day. They are key to innovation in critical fields like medicine, electronics, and sustainability. The search for the next generation of groundbreaking, eco-friendly materials is on, and machine learning can be the solution. However, progress has been stalled by one major hurdle: a critical lack of accessible, high-quality data.

Our Open Polymer Prediction 2025 introduces a game-changing, large-scale open-source dataset—ten times larger than any existing resource. We invite you to piece together the missing links and unlock the vast potential of sustainable materials.

Your mission is to predict a polymer's real-world performance directly from its chemical structure. You will be provided with a polymer's structure as a simple text string (SMILES), and your challenge is to build a model that can accurately forecast five key metrics that determine how it will behave. These include predicting:

- **Glass transition temperature** (`Tg`)
- **Fractional free volume** (`FFV`)
- **Thermal conductivity** (`Tc`)
- **Polymer density** (`Density`)
- **Radius of gyration** (`Rg`)

The ground truth for this competition is averaged from multiple runs of molecular dynamics simulation. Successfully predicting these properties is crucial for scientists to accelerate the design of novel polymers with targeted characteristics, which can be used in various applications.

### Evaluation Metric & Interface

- **Evaluation Metric**: Combination of weighted MAE and R^2  
- **Interface File**: `deepevolve_interface.py`

The weighted Mean Absolute Error (wMAE) is defined as:

```math
\mathrm{wMAE} = \frac{1}{\lvert \mathcal{X} \rvert} \sum_{X \in \mathcal{X}} \sum_{i \in I(X)} w_{i}\,\bigl\lvert \hat{y}_{i}(X) - y_{i}(X) \bigr\rvert
```

Each property is assigned a weight $w_{i}$ to ensure that all property types contribute equally regardless of their scale or frequency:

$$
w_{i} = \frac{1}{r_{i}} \; \times \; \frac{K\,\sqrt{\tfrac{1}{n_{i}}}}{\displaystyle\sum_{j=1}^{K}\sqrt{\tfrac{1}{n_{j}}}}
$$

### Data Files

The dataset is provided in three CSV files: `train.csv`, `valid.csv`, and `test.csv`. Each file contains the following columns:

| Column    | Description                                              |
|-----------|----------------------------------------------------------|
| `id`      | Unique identifier for each polymer.                    |
| `SMILES`  | Sequence-like chemical notation of polymer structures. |
| `Tg`      | Glass transition temperature (°C).                     |
| `FFV`     | Fractional free volume.                                |
| `Tc`      | Thermal conductivity (W/m·K).                          |
| `Density` | Polymer density (g/cm³).                                 |
| `Rg`      | Radius of gyration (Å).                                  |

## Initial Idea

The initial approach leverages insights from the paper "[Graph Rationalization with Environment-based Augmentations](https://arxiv.org/abs/2206.02886)". This method focuses on enhancing the representation of polymer structures by applying environment-based augmentations to graph representations. The idea is to rationalize the graph structure, ensuring that key chemical and physical interactions are preserved and highlighted during model training.

For more details and supplementary code, please refer to the GitHub repository for GREA: [GREA Repository](https://github.com/liugangcode/GREA). This approach aims to improve model interpretability and performance, and it offers a robust framework for tackling complex chemical property prediction tasks in polymer science.