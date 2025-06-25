import numpy as np
import pandas as pd
import sys
from sklearn.utils.validation import check_consistent_length
from data_loader import load_data, preprocess_supplement_data
from preprocessing import DataPrep
from config import LGB_PARAMS, get_nn_config, get_lgb_features, get_nn_features
from lightgbm_model import LGBClassModel1
from neural_network import NNRegModel1
from utils import repl
from public_timeseries_testing_util import MockApi


def smapep1(y_true, y_pred):
    """SMAPE of y+1, a nonnegative float, smaller is better

    Parameters: y_true, y_pred: array-like

    Returns 100 for 100 % error.
    y_true may have missing values.
    """
    check_consistent_length(y_true, y_pred)
    y_true = np.array(y_true, copy=False).ravel()
    y_pred = np.array(y_pred, copy=False).ravel()
    y_true, y_pred = y_true[np.isfinite(y_true)], y_pred[np.isfinite(y_true)]
    if (y_true < 0).any():
        raise ValueError("y_true < 0")
    if (y_pred < 0).any():
        raise ValueError("y_pred < 0")
    denominator = (y_true + y_pred) / 2 + 1
    ape = np.abs(y_pred - y_true) / denominator
    return np.average(ape) * 100


def main_func(base_dir):
    proteins, peptides, clinical, supplement = load_data(base_dir)
    supplement = preprocess_supplement_data(supplement)

    # Initialize data preprocessor
    dp3 = DataPrep()
    dp3.fit(proteins, peptides, clinical)

    # Prepare training samples
    sample3 = dp3.transform_train(proteins, peptides, clinical)
    sample3 = sample3[~sample3["target"].isnull()]
    sample3["is_suppl"] = 0

    sup_sample3 = dp3.transform_train(proteins, peptides, supplement)
    sup_sample3 = sup_sample3[~sup_sample3["target"].isnull()]
    sup_sample3["is_suppl"] = 1

    # Train LightGBM model
    lgb_features = get_lgb_features()
    model_lgb = LGBClassModel1(LGB_PARAMS, lgb_features)
    model_lgb = model_lgb.fit(pd.concat([sample3, sup_sample3], axis=0))

    # Train Neural Network model
    cfg = get_nn_config()
    cfg.features = get_nn_features(sample3)
    model_nn = NNRegModel1(cfg)
    model_nn = model_nn.fit(pd.concat([sample3, sup_sample3], axis=0))

    # Load test environment (if available)
    env = MockApi(base_dir)
    iter_test = env.iter_test()

    all_test_peptides = None
    all_test_proteins = None
    all_test_df = None

    for test_df, test_peptides, test_proteins, sample_submission in iter_test:

        all_test_df = pd.concat([all_test_df, test_df], axis=0)
        all_test_proteins = pd.concat([all_test_proteins, test_proteins], axis=0)
        all_test_peptides = pd.concat([all_test_peptides, test_peptides], axis=0)

        sample_test = dp3.transform_test(
            all_test_proteins, all_test_peptides, all_test_df, sample_submission
        )
        sample_test["is_suppl"] = 0

        if not sample_test.empty:
            sample_test["preds_lgb"] = model_lgb.predict(sample_test)
            sample_test["preds_nn"] = np.round(
                np.clip(model_nn.predict(sample_test), 0, None)
            )
            sample_submission["rating"] = np.round(
                (sample_test["preds_lgb"] + sample_test["preds_nn"]) / 2
            )

        env.predict(sample_submission)

    # Read final submission
    prediction = env.get_predictions()
    solution = env.get_answer()
    score = smapep1(solution["rating"], prediction["rating"])
    return score


if __name__ == "__main__":
    base_dir = "../../../data_cache/amp_pd"
    score = main_func(base_dir)
    print("score", score)