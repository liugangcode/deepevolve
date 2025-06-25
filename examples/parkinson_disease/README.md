# Parkinson's Disease Progression Prediction Competition

This repository contains the code and documentation for the Parkinson's Disease progression prediction competition. The goal is to forecast MDS-UPDRS scores, which measure both motor and non-motor symptoms in Parkinson's patients, using clinical and molecular data. This document outlines the problem statement, dataset details, evaluation metric, and an overview of the winning solution.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Evaluation Metric](#evaluation-metric)
- [Competition Interface](#competition-interface)
- [1st Place Solution](#1st-place-solution)
- [References](#references)

---

## Overview

Parkinson's disease (PD) is a progressive neurological disorder affecting movement, cognition, and other functions. With an increasing number of people predicted to be affected in the future, using data science to understand and predict disease progression could pave the way for new treatments. The competition focuses on developing a model to predict MDS-UPDRS scores using a combination of clinical assessments and protein abundance data.

---

## Problem Statement

The objective is to predict MDS-UPDRS scores for patient visits and to forecast the scores for subsequent visits (6, 12, and 24 months later) using historical clinical data and molecular (protein and peptide) information. Your model will help uncover potential biomarkers responsible for disease progression, aiding in the development of novel pharmacotherapies.

### Key Points

- **Clinical Focus:** MDS-UPDRS (Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale) scores.
- **Data Types:** Protein abundance data (from cerebrospinal fluid samples) and clinical data.
- **Prediction Horizon:** Forecast scores for current visits as well as future visits (6, 12, and 24 months).
- **Competition Type:** This is a Code Competition with strict requirements on runtime (under five minutes) and memory usage (less than 0.5 GB).

---

## Dataset Description

The core dataset comprises several CSV files containing detailed information from both clinical assessments and protein mass spectrometry. Below is a brief overview of each file:

- **train_peptides.csv**  
  Contains peptide-level mass spectrometry data.  
  **Columns:**  
  - `visit_id`: Unique identifier for the patient visit.  
  - `visit_month`: Month of the visit relative to the patient's first visit.  
  - `patient_id`: Unique patient identifier.  
  - `UniProt`: Protein identifier.  
  - `Peptide`: Amino acid sequence of the peptide.  
  - `PeptideAbundance`: Frequency of the peptide in the sample.

- **train_proteins.csv**  
  Aggregated protein expression data from the peptide-level information.  
  **Columns:**  
  - `visit_id`, `visit_month`, `patient_id`, `UniProt`  
  - `NPX`: Normalized protein expression value.

- **train_clinical_data.csv**  
  Clinical assessments including UPDRS scores.  
  **Columns:**  
  - `visit_id`, `visit_month`, `patient_id`  
  - `updrs_1` to `updrs_4`: Different aspects of the UPDRS score.  
  - `upd23b_clinical_state_on_medication`: Indicator of medication usage during the UPDRS assessment.

- **supplemental_clinical_data.csv**  
  Additional clinical records without corresponding CSF samples.

- **example_test_files/**  
  Sample data files demonstrating how the API delivers test set data (note that UPDRS columns are omitted).

- **amp_pd_peptide/**  
  Files to support API functionality and ensure that predictions can be generated efficiently.

- **public_timeseries_testing_util.py**  
  An optional utility for running custom offline tests using the time-series API.

---

## Evaluation Metric

Submissions are evaluated using the following metric:

```math
\frac{1 - \mathrm{SMAPE}(\%)}{2}
```

where SMAPE is the Symmetric Mean Absolute Percentage Error. For cases where both actual and predicted values are zero, SMAPE is defined as 0.

During evaluation, for each patient visit where a protein/peptide sample was captured, you are required to:
- Estimate the UPDRS scores for that visit.
- Predict the scores for potential future visits at 6, 12, and 24 months.  
Predictions for visits that did not occur are ignored.

---

## Competition Interface

The competition utilizes the `deepevolve_interface.py` file to manage submissions. Your solution should be optimized to work within the following constraints:
- **Runtime:** Less than five minutes for the complete dataset.
- **Memory:** Less than 0.5 GB usage.

Submissions should adhere to the code requirements provided in the competition guidelines.

---

## 1st Place Solution

The winning solution was based on a simple averaging of two models: LightGBM (LGB) and Neural Network (NN). Both models shared the same features, with minor differences in preprocessing. Below is a brief overview of the key components:

### Features Used
- Visit month  
- Forecast horizon  
- Target prediction month  
- Indicator of blood sample collection during the visit  
- Indicator from the supplementary dataset  
- Binary indicators for visits occurring at 6th, 18th, and 48th months  
- Count of previous non-annual visits (e.g., 6th or 18th month visits)  
- Index of the target (after pivoting the dataset)

Notably, the final solution discarded the blood test results after thorough experimentation as these features did not improve model performance.

### LGB Model
- **Approach:**  
  Instead of a traditional regression, the LGB model was re-framed as a classification problem with 87 target classes (ranging from 0 to the maximum target value).  
- **Objective:**  
  Log loss was used as the objective function, and a custom post-processing step was implemented. This step selects the predicted value minimizing $ \mathrm{SMAPE} + 1 $ across the predicted distribution, effectively searching among 87 possible integer values.
- **Hyperparameter Tuning:**  
  An optimization routine was used to fine-tune model hyperparameters to minimize the target metric.

### Neural Network
- **Architecture:**  
  A multi-layer feed-forward network designed for regression, directly optimizing the $ \mathrm{SMAPE} + 1 $ loss.
- **Activation Function:**  
  A leaky ReLU was employed as the final activation to prevent negative predictions, ensuring that the outputs remain clinically sensible.

### Cross-Validation
- **Strategy:**  
  Several cross-validation schemes were considered, with the final model using a leave-one-patient-out (group k-fold) strategy.  
- **Observations:**  
  The cross-validation scores were well aligned with the private leaderboard scores, validating the effectiveness of the chosen strategy.

### Insights
- The presence of a visit at the 6th month was highly predictive and correlated strongly with higher UPDRS scores.
- Predictions for forecasts made during `visit_month = 0` exhibited a systematic pattern, which was adjusted to better reflect clinical reasoning.
- Efforts to incorporate blood test data did not yield improvements. Multiple modeling approaches using these features were attempted, but none improved cross-validation performance reliably. Less complex versions of these features were later introduced with minor effects on public leaderboard performance.

For more details and implementation specifics, please refer to the original Kaggle kernel:

[1st Place Solution on Kaggle](https://www.kaggle.com/code/dott1718/1st-place-solution?scriptVersionId=129798049)