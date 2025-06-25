import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from metrics import smape1p


def split_df(df, folds_mapping, fold_id: int = 0):
    """Split dataframe into train and validation sets."""
    folds = df["patient_id"].map(folds_mapping)

    df_train = df[folds != fold_id]
    df_train = df_train[~df_train["target"].isnull()].reset_index(drop=True)

    df_valid = df[folds == fold_id]
    df_valid = df_valid[~df_valid["target"].isnull()].reset_index(drop=True)

    return df_train, df_valid


def create_folds_mapping(df, n_folds=5, random_state=42):
    """Create patient-level fold mapping."""
    folds_df = pd.DataFrame({"patient_id": df["patient_id"].unique()})
    folds_df["fold"] = -1

    for i, (_, test_index) in enumerate(
        KFold(n_splits=n_folds, shuffle=True, random_state=random_state).split(folds_df)
    ):
        folds_df.loc[test_index, "fold"] = i
    folds_mapping = folds_df.set_index(["patient_id"])["fold"]
    return folds_mapping


def run_single_fit(model, df_train, df_valid, fold_id, seed, probs):
    """Run a single model fit and prediction."""
    if probs:
        p = model.fit_predict_proba(df_train, df_valid)
        p = pd.DataFrame(
            p, columns=[f"prob_{i}" for i in range(p.shape[1])]
        ).reset_index(drop=True)
        res = pd.DataFrame(
            {
                "seed": seed,
                "fold": fold_id,
                "patient_id": df_valid["patient_id"],
                "visit_month": df_valid["visit_month"],
                "target_month": df_valid["target_month"],
                "target_i": df_valid["target_i"],
                "target": df_valid["target"],
            }
        ).reset_index(drop=True)
        return pd.concat([res, p], axis=1)
    else:
        p = model.fit_predict(df_train, df_valid)
        return pd.DataFrame(
            {
                "seed": seed,
                "fold": fold_id,
                "patient_id": df_valid["patient_id"],
                "visit_month": df_valid["visit_month"],
                "target_month": df_valid["target_month"],
                "target_i": df_valid["target_i"],
                "target": df_valid["target"],
                "preds": p,
            }
        )


class BaseModel:
    """Base class for all models."""

    def fit(self, df_train):
        raise NotImplementedError

    def predict(self, df_valid):
        raise NotImplementedError

    def predict_proba(self, df_valid):
        raise NotImplementedError

    def fit_predict(self, df_train, df_valid):
        self.fit(df_train)
        return self.predict(df_valid)

    def fit_predict_proba(self, df_train, df_valid):
        self.fit(df_train)
        return self.predict_proba(df_valid)

    def cv(self, sample, sup_sample=None, n_folds=5, random_state=42):
        """Cross-validation."""
        folds_mapping = create_folds_mapping(sample, n_folds, random_state)

        res = None
        for fold_id in sorted(folds_mapping.unique()):
            df_train, df_valid = split_df(sample, folds_mapping, fold_id)
            if sup_sample is not None:
                df_train = pd.concat([df_train, sup_sample], axis=0)
            p = self.fit_predict(df_train, df_valid)
            delta = pd.DataFrame(
                {
                    "fold": fold_id,
                    "patient_id": df_valid["patient_id"],
                    "visit_month": df_valid["visit_month"],
                    "target_month": df_valid["target_month"],
                    "target_i": df_valid["target_i"],
                    "target": df_valid["target"],
                    "preds": p,
                }
            )
            res = pd.concat([res, delta], axis=0)

        return res

    def cvx(
        self, sample, sup_sample=None, n_runs=1, n_folds=5, random_state=42, probs=False
    ):
        """Extended cross-validation with multiple runs."""
        np.random.seed(random_state)
        seeds = np.random.randint(0, 1e6, n_runs)

        run_args = []
        for seed in seeds:
            folds_mapping = create_folds_mapping(sample, n_folds, seed)
            for fold_id in sorted(folds_mapping.unique()):
                df_train, df_valid = split_df(sample, folds_mapping, fold_id)
                if sup_sample is not None:
                    df_train = pd.concat([df_train, sup_sample], axis=0)
                run_args.append(
                    dict(
                        df_train=df_train,
                        df_valid=df_valid,
                        fold_id=fold_id,
                        seed=seed,
                        probs=probs,
                    )
                )

        res = Parallel(-1)(delayed(run_single_fit)(self, **args) for args in run_args)
        return pd.concat(res, axis=0)

    def loo(self, sample, sup_sample=None, probs=False, sample2=None):
        """Leave-one-out cross-validation."""
        if sample2 is None:
            sample2 = sample
        run_args = []
        for patient_id in sample["patient_id"].unique():
            df_train = sample[sample["patient_id"] != patient_id]
            df_valid = sample2[sample2["patient_id"] == patient_id]
            if sup_sample is not None:
                df_train = pd.concat([df_train, sup_sample], axis=0)
            run_args.append(
                dict(
                    df_train=df_train,
                    df_valid=df_valid,
                    fold_id=None,
                    seed=None,
                    probs=probs,
                )
            )

        res = Parallel(-1)(delayed(run_single_fit)(self, **args) for args in run_args)
        return pd.concat(res, axis=0)


