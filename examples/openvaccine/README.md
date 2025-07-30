# OpenVaccine Competition

Welcome to the OpenVaccine competition repository. This competition challenges participants to develop models to predict RNA degradation rates at the base level, a critical step towards designing stable mRNA vaccines for COVID-19.

---

## Overview

### Competition Background

Winning the fight against the COVID-19 pandemic requires an effective vaccine that can be equitably and widely distributed. mRNA vaccines show significant promise but face challenges related to stability. RNA molecules are prone to degradation and even a single cut can render the vaccine ineffective. Understanding the degradation mechanisms at the base level is therefore essential.

The Eterna community, in collaboration with researchers from Stanford University, has embraced a novel approach. By combining scientific expertise with crowdsourced insights from gamers, they tackle challenges in RNA design. This competition leverages data science to design models predicting degradation rates at each RNA base position, aiding the acceleration of mRNA vaccine development.

### Data Overview

Participants will work with a dataset comprised of over 3000 RNA molecules, where each molecule is annotated with a series of experimental measurements. The dataset includes:

- **train.json** – Contains the training data.
- **test.json** – Test data without any ground truth values.
- **sample_submission.csv** – A sample submission file demonstrating the required format.

Each sample includes the following columns:

- **id** – Unique identifier for each sample.
- **seq_scored** – Number of positions used in scoring. In Train and Public Test the value is 68, while it is 91 in the Private Test.
- **seq_length** – Length of the RNA sequence. This is 107 for Train and Public Test, and 130 for Private Test.
- **sequence** – RNA sequence comprising the characters A, G, U, and C.
- **structure** – Dot-bracket notation representing the secondary structure of the RNA.
- **reactivity, deg_pH10, deg_Mg_pH10, deg_50C, deg_Mg_50C** – Vectors of experimental measurements (reactivity and different degradation rates) for positions specified by **seq_scored**.
- **\*_error_\*** – Estimated errors corresponding to the experimental measurements.
- **predicted_loop_type** – Structural context (or loop type) assigned to each base. The loop types include:
  - S: Stem (paired)
  - M: Multiloop
  - I: Internal loop
  - B: Bulge
  - H: Hairpin loop
  - E: Dangling end
  - X: External loop
- **S/N_filter** – Indicator if the sample passed additional quality filters.

**Test Set Filtering:**  
The 629 RNA sequences in the test set were selected based on the following criteria:
- A minimum value greater than -0.5 across all 5 conditions.
- A signal-to-noise ratio (mean measurement over 68 nts divided by mean statistical error) greater than 1.0.
- Less than 50% sequence similarity within clusters containing at most 3 similar sequences.

*Note: The public training data contains additional noisy measurements to allow competitors to extract further insights.*

---

## Evaluation Metric

Submissions are scored using the mean columnwise root mean squared error (MCRMSE):

```math
\mathrm{MCRMSE} = \frac{1}{N_t}\sum_{j=1}^{N_t}\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_{ij}-\hat{y}_{ij})^2},
```

where:
- $N_t$ is the number of ground-truth target columns that are scored.
- $n$ is the number of samples.
- $y_{ij}$ and $\hat{y}_{ij}$ are the actual and predicted values, respectively.

Although the training data provides five ground-truth measurements, only the following three are used for scoring:
- **reactivity**
- **deg_Mg_pH10**
- **deg_Mg_50C**

---

## Proposed Approach

### Model: GraphSAGE (with GCN) + GRU + KFold

This repository presents a model that integrates GraphSAGE-based graph convolution with a GRU and k-fold cross-validation framework. The approach is designed as follows:

1. **Feature Embedding:**  
   Each nucleotide is embedded along with its predicted secondary structure and loop-type context.

2. **Graph Construction:**  
   A graph is constructed where:
   - Nodes represent individual RNA bases.
   - Edges connect adjacent bases and bases that are paired in the structure.

3. **Graph Convolution:**  
   A GraphSAGE-based convolution network aggregates information from neighboring nodes, yielding enriched base-level features.

4. **Sequence Modeling:**  
   The enriched features are passed through a bidirectional GRU to capture sequential patterns along the RNA chain.

5. **Prediction:**  
   A linear output layer predicts the three scored targets (reactivity, deg_Mg_pH10, and deg_Mg_50C) for each position in the RNA sequence.

6. **Cross-Validation:**  
   Training employs k-fold cross-validation to ensure robust performance across the public dataset.

### Optional Enhancements

An optional enhancement involves the incorporation of precomputed base-pair probability (bpps) matrices. These matrices provide a richer view of the RNA folding ensemble and can be used as additional node features. Participants can choose to integrate them into the graph or disregard them.

For more details and a complete implementation, please refer to the accompanying Kaggle notebook:  
[GraphSAGE (Graph Convolution) & GRU with KFold Implementation](https://www.kaggle.com/code/vudangthinh/openvaccine-gcn-graphsage-gru-kfold/notebook#Pytorch-model-based-on-GCN-(GraphSAGE)-and-GRU)