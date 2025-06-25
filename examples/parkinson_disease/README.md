# Parkinson Disease

## Problem Description

The goal of this competition is to predict MDS-UPDR scores, which measure progression in patients with Parkinson's disease. The Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS) is a comprehensive assessment of both motor and non-motor symptoms associated with Parkinson's. Participants will develop a model trained on data of protein and peptide levels over time in subjects with Parkinson's disease versus normal age-matched control subjects.

Parkinson's disease (PD) is a disabling brain disorder that affects movements, cognition, sleep, and other normal functions. Unfortunately, there is no current cure, and the disease worsens over time. It's estimated that by 2037, 1.6 million people in the U.S. will have Parkinson's disease, at an economic cost approaching $80 billion. Research indicates that protein or peptide abnormalities play a key role in the onset and worsening of this disease. Gaining a better understanding of this—with the help of data science—could provide important clues for the development of new pharmacotherapies to slow the progression or cure Parkinson's disease.

The Accelerating Medicines Partnership® Parkinson's Disease (AMP®PD) is a public-private partnership between government, industry, and nonprofits managed through the Foundation of the National Institutes of Health (FNIH). The Partnership created the AMP PD Knowledge Platform, which includes a deep molecular characterization and longitudinal clinical profiling of Parkinson's disease patients, with the goal of identifying and validating diagnostic, prognostic, and/or disease progression biomarkers for Parkinson's disease.

Submissions are evaluated on the Symmetric Mean Absolute Percentage Error (SMAPE) between forecasts and actual values. SMAPE is defined as:

\[ \text{SMAPE} = \frac{100\%}{n} \sum_{t=1}^{n} \frac{|F_t - A_t|}{(|A_t| + |F_t|)/2} \]

where \( F_t \) is the forecasted value and \( A_t \) is the actual value. SMAPE is set to 0 when both actual and predicted values are 0.

For each patient visit where a protein/peptide sample was taken, you will need to estimate both their UPDRS scores for that visit and predict their scores for any potential visits 6, 12, and 24 months later. Predictions for any visits that didn't ultimately take place are ignored.

- **Evaluation Metric**: \((1-\text{SMAPE}(\%))/2\)
- **Interface File**: `deepevolve_interface.py`

## Initial Idea

### 1st Place Solution

Our final solution is a simple average of two models: LightGBM (LGB) and a Neural Network (NN). Both models were trained on the same features, with additional scaling and binarization for the NN:

- Visit month
- Forecast horizon
- Target prediction month
- Indicator whether blood was taken during the visit
- Supplementary dataset indicator
- Indicators whether a patient visit occurred on the 6th, 18th, and 48th month
- Count of number of previous "non-annual" visits (6th or 18th)
- Index of the target (we pivot the dataset to have a single target column)

The winning solution fully ignores the results of the blood tests. Despite extensive efforts to find any signal in this crucial piece of data, none of our approaches or models could benefit from blood test features significantly enough to distinguish them from random variations. The final models were trained only on the union of clinical and supplementary datasets.

#### LightGBM (LGB)

Throughout the competition, LGB was our model to beat, and only a NN trained with the competition metric as the loss function was able to achieve competitive performance on cross-validation. Initially, we experimented with a regression LGB model using different hyperparameters and custom objective functions, but nothing surpassed L1 regression, which does not optimize the desired metric SMAPE+1. We observed that the performance of every model on cross-validation improved when the regression outputs were rounded to integers. Consequently, we adopted an alternative approach.

Our LGB model is a classification model with 87 target classes (0 to maximum target value) and a logloss objective. To produce the forecast, we applied the following post-processing: given the predicted distribution of target classes, we selected the value that minimizes SMAPE+1. Observing that the optimal predictions are always integers, the task reduced to a trivial search among 87 possible values. This approach naturally handles cases with multiple local minima and would also work for the original SMAPE metric.

We ran an optimization routine to tune LGB hyperparameters to minimize SMAPE+1 on cross-validation using the described post-processing.

#### Neural Network (NN)

The neural network has a simple multi-layer feed-forward architecture with a regression target, using the competition metric SMAPE+1 as the loss function. We fixed the number of epochs and scheduler, and then tuned the learning rate and hidden layer size. The only trick was to add a leaky ReLU activation as the last layer to prevent the NN from producing negative predictions. There are alternative ways to handle this issue.

#### Cross-Validation

Due to the small training sample size, we experimented with multiple cross-validation schemes, all stratified by patient ID. Once a sufficient number of folds was used, they correlated well with each other and with the private leaderboard. The final scheme was leave-one-patient-out, or group k-fold cross-validation with one fold per patient, which does not depend on random numbers. The resulting cross-validation scores aligned well with the private leaderboard, and our chosen submission was our best on the private leaderboard.

#### What Worked

The most impactful feature was the indicator of a visit on the 6th month. It correlates strongly with UPDRS targets (especially parts 2 and 3) and with medication frequency. We observed that patients who returned at 6 months tend to have higher UPDRS scores on average. A similar effect exists for the 18th month visit, but these features are correlated. 

Another effect is seen for forecasts at visit_month = 0: forecasts for 0, 12, and 24 months ahead are consistently lower than for 6 months ahead. Mathematically, this makes sense because if a patient returns at 6 months they have higher UPDRS scores on average, and if not, the forecast is ignored. Clinically, however, this behavior is unreasonable.

It was also important to note differences between training and test datasets, which explain why adding a feature for a 30th month visit might improve cross-validation but harm leaderboard performance.

#### What Did Not Work

Blood test data. We tried many methods to incorporate proteins and peptides into our models, but none improved cross-validation. We narrowed it to a bag of logistic regressions predicting a 6th month visit from the 0th month blood test. We applied soft up/down scaling of predictions based on these probabilities, which improved the public leaderboard after tuning a few coefficients directly on it. That approach reached second place on the public leaderboard but clearly overfit. We included a mild version of it in our second final submission, which scored slightly worse on the private leaderboard (60.0 vs. 60.3).

For more details, please refer to the [1st Place Solution](https://www.kaggle.com/code/dott1718/1st-place-solution?scriptVersionId=129798049).