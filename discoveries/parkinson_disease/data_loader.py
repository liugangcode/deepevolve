import pandas as pd


def load_data(base_path="data"):
    """Load training data from CSV files."""
    proteins = pd.read_csv(f"{base_path}/train_proteins.csv")
    peptides = pd.read_csv(f"{base_path}/train_peptides.csv")
    clinical = pd.read_csv(f"{base_path}/train_clinical_data.csv")
    supplement = pd.read_csv(f"{base_path}/supplemental_clinical_data.csv")
    return proteins, peptides, clinical, supplement


def preprocess_supplement_data(supplement_df):
    """Preprocess supplement data."""
    supplement_df.loc[supplement_df["visit_month"] == 5, "visit_month"] = 6
    return supplement_df


