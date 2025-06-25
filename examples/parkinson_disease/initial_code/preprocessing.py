import numpy as np
import pandas as pd
from config import TARGET_HORIZONS, TEST_VMONTHS


class DataPrep:
    def __init__(self, target_horizons=None, test_vmonths=None):
        self.target_horizons = target_horizons or TARGET_HORIZONS
        self.test_vmonths = test_vmonths or TEST_VMONTHS

    def fit(self, proteins_df, peptides_df, clinical_df):
        """Fit the data preprocessor (placeholder for future extensions)."""
        pass

    def fe(self, sample, proteins_df, peptides_df, clinical_df):
        """Feature engineering."""
        # Visit month features
        for v_month in [0, 6, 12, 18, 24, 36, 48, 60, 72, 84]:
            p = list(
                clinical_df[clinical_df["visit_month"] == v_month][
                    "patient_id"
                ].unique()
            )
            sample[f"visit_{v_month}m"] = sample.apply(
                lambda x: (x["patient_id"] in p) and (x["visit_month"] >= v_month),
                axis=1,
            ).astype(int)

            p = list(
                proteins_df[proteins_df["visit_month"] == v_month][
                    "patient_id"
                ].unique()
            )
            sample[f"btest_{v_month}m"] = sample.apply(
                lambda x: (x["patient_id"] in p) and (x["visit_month"] >= v_month),
                axis=1,
            ).astype(int)

            sample[f"t_month_eq_{v_month}"] = (
                sample["target_month"] == v_month
            ).astype(int)
            sample[f"v_month_eq_{v_month}"] = (sample["visit_month"] == v_month).astype(
                int
            )

        # Horizon features
        for hor in self.target_horizons:
            sample[f"hor_eq_{hor}"] = (sample["horizon"] == hor).astype(int)

        sample["horizon_scaled"] = sample["horizon"] / 24.0

        # Blood test features
        blood_samples = proteins_df["visit_id"].unique()
        sample["blood_taken"] = sample.apply(
            lambda x: x["visit_id"] in blood_samples, axis=1
        ).astype(int)

        # Visit count features
        all_visits = (
            clinical_df.groupby("patient_id")["visit_month"]
            .apply(lambda x: list(set(x)))
            .to_dict()
        )
        all_non12_visits = sample.apply(
            lambda x: [
                xx
                for xx in all_visits.get(x["patient_id"], [])
                if xx <= x["visit_month"] and xx % 12 != 0
            ],
            axis=1,
        )
        sample["count_non12_visits"] = all_non12_visits.apply(lambda x: len(x))

        return sample

    def transform_train(self, proteins_df, peptides_df, clinical_df):
        """Transform training data."""
        sample = clinical_df.rename(
            {"visit_month": "target_month", "visit_id": "visit_id_target"}, axis=1
        ).merge(
            clinical_df[["patient_id", "visit_month", "visit_id"]],
            how="left",
            on="patient_id",
        )

        sample["horizon"] = sample["target_month"] - sample["visit_month"]
        sample = sample[sample["horizon"].isin(self.target_horizons)]
        sample = sample[sample["visit_month"].isin(self.test_vmonths)]

        # Features
        sample = self.fe(
            sample,
            proteins_df[proteins_df["visit_month"].isin(self.test_vmonths)],
            peptides_df[peptides_df["visit_month"].isin(self.test_vmonths)],
            clinical_df[clinical_df["visit_month"].isin(self.test_vmonths)],
        )

        # Targets reshape
        res = []
        for tgt_i in np.arange(1, 5):
            delta_df = sample.copy()
            if f"updrs_{tgt_i}" in delta_df.columns:
                delta_df["target"] = delta_df[f"updrs_{tgt_i}"]
                delta_df["target_norm"] = delta_df["target"] / 100
            delta_df["target_i"] = tgt_i
            res.append(delta_df)

        sample = pd.concat(res, axis=0).reset_index(drop=True)
        if f"updrs_1" in sample.columns:
            sample = sample.drop(["updrs_1", "updrs_2", "updrs_3", "updrs_4"], axis=1)

        for tgt_i in np.arange(1, 5):
            sample[f"target_n_{tgt_i}"] = (sample["target_i"] == tgt_i).astype(int)

        return sample

    def transform_test(self, proteins_df, peptides_df, test_df, sub_df):
        """Transform test data."""
        sub = sub_df.copy()
        sub["patient_id"] = sub["prediction_id"].apply(lambda x: int(x.split("_")[0]))
        sub["visit_month"] = sub["prediction_id"].apply(lambda x: int(x.split("_")[1]))
        sub["visit_id"] = sub.apply(
            lambda x: str(x["patient_id"]) + "_" + str(x["visit_month"]), axis=1
        )

        sample = sub[["patient_id", "visit_month", "visit_id", "prediction_id"]]

        sample["horizon"] = sample["prediction_id"].apply(
            lambda x: int(x.split("_")[5])
        )
        sample["target_i"] = sample["prediction_id"].apply(
            lambda x: int(x.split("_")[3])
        )
        sample["visit_month"] = sample["visit_month"]
        sample["target_month"] = sample["visit_month"] + sample["horizon"]
        del sample["prediction_id"]

        # Features
        sample = self.fe(sample, proteins_df, peptides_df, test_df)

        for tgt_i in np.arange(1, 5):
            sample[f"target_n_{tgt_i}"] = (sample["target_i"] == tgt_i).astype(int)

        return sample
