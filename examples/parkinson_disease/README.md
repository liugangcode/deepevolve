# parkinson_disease

## Problem Description

The goal of this competition is to predict MDS-UPDR scores, which measure the progression of Parkinson's disease in patients. The MDS-UPDRS (Movement Disorder Society-Sponsored Revision of the Unified Parkinson’s Disease Rating Scale) is a comprehensive assessment tool that examines both motor and non-motor symptoms associated with Parkinson's disease. Participants are expected to develop a model trained on time-series data of protein and peptide levels obtained from subjects with Parkinson's disease and normal age-matched controls.  
  
Your work could provide crucial insights into which molecules change as Parkinson's disease progresses, potentially contributing to breakthrough research and new pharmacotherapies.

### Context
Parkinson's disease (PD) is a disabling brain disorder that affects movement, cognition, sleep, and other functions. There is currently no cure, and the disease worsens over time. It is estimated that by 2037, 1.6 million people in the U.S. will suffer from Parkinson's disease, incurring economic costs approaching \$80 billion. Abnormalities in proteins or peptides are believed to play a key role in the onset and progression of PD. This challenge leverages clinical data and mass spectrometry-based protein abundance data to predict disease progression using time-series predictions.

### Dataset Description
The dataset comprises measurements from cerebrospinal fluid (CSF) samples, including:
- **train_peptides.csv**: Mass spectrometry data at the peptide level with details such as visit ID, relative visit month, patient ID, UniProt ID, peptide sequence, and peptide abundance.
- **train_proteins.csv**: Aggregated protein expression frequencies derived from peptides with columns for visit ID, visit month, patient ID, UniProt ID, and normalized protein expression (NPX).
- **train_clinical_data.csv**: Clinical assessments including visit ID, month, patient ID, UPDRS scores (parts 1 to 4), and medication status during assessment.
- **supplemental_clinical_data.csv**: Additional clinical records without associated CSF samples for broader context.
- Additional files and folders provide example test files, API enabling files, and a utility for running custom offline API tests.

### Competition Specifics
- **Time-Series Nature**: Predictions should estimate the current UPDRS score at the visit and predict scores 6, 12, and 24 months later. For any visits that did not occur, predictions will be ignored.
- **Code Competition**: Please refer to the Code Requirements for additional details.

- **Evaluation Metric**: (1-SMAPE(%))/2, where SMAPE is the Symmetric Mean Absolute Percentage Error (SMAPE or sMAPE)
- **Interface File**: `deepevolve_interface.py`

---

## Initial Idea

### 1st Place Solution

The winning approach was built on a simple ensemble that averages predictions from two models: LightGBM (LGB) and a Neural Network (NN). Both models were trained using the same set of features with slight preprocessing differences. The main features include:
- Visit month
- Forecast horizon and target prediction month
- An indicator for whether blood was taken during the visit
- A supplementary dataset indicator
- Boolean indicators for patient visits on the 6th, 18th, and 48th month
- Count of previous “non-annual” visits (specifically, visits at 6 or 18 months)
- An index representing the pivoted target column

#### LightGBM (LGB)
- **Approach**: Initially deployed as a regression model with various hyperparameters and custom objective functions, it was later transformed into a classification model. The classification model predicts among 87 discrete target classes (ranging from 0 to the maximum target value) using a log loss objective.
- **Post-Processing**: The predicted probability distribution over classes is used to select the target value that minimizes the SMAPE+1 metric. This approach naturally handles cases with multiple local minima.
- **Hyperparameter Tuning**: An optimization routine was run to tune the LGB hyperparameters to minimize SMAPE+1 on cross-validation.

#### Neural Network (NN)
- **Architecture**: A simple multi-layer feed-forward network configured for regression, with SMAPE+1 as the loss function.
- **Key Adjustments**: 
  - Fixed number of epochs and scheduler.
  - Tuned the learning rate and hidden layer size.
  - Applied a leaky ReLU activation at the final layer to prevent negative predictions.
  
#### Cross-Validation Strategy
Due to the limited training sample size, several cross-validation schemes were explored. The final method was a leave-one-patient-out (group k-fold with one fold per patient) strategy, which proved to correlate well with the private leaderboard results.

#### Observations and Feature Impact
- **Effective Features**: An indicator for a visit on the 6th month had the most significant impact, correlating strongly with UPDRS parts (especially parts 2 and 3) and medication frequency.
- **Model Refinements**: Predictions for visits at month 0 showed consistently lower forecasts for longer horizons. While this made sense statistically, it raised questions regarding clinical expectations.
- **Failed Approaches**: Extensive efforts to integrate blood test data (protein and peptide features) did not yield measurable improvements, leading to a final strategy that focused on clinical and supplementary datasets.

For a detailed walkthrough of the winning solution, please refer to the [1st Place Solution](https://www.kaggle.com/code/dott1718/1st-place-solution?scriptVersionId=129798049).

---

MathJax is enabled for equations. For example, the evaluation metric can be visualized as:  
$$
\text{Evaluation Metric} = \frac{1 - \text{SMAPE}(\%)}{2}
$$

This completes the documentation for the parkinson_disease challenge.