import os
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from time import time

def download_raw_data(taskname: str, download_dir: str = "./polymer"):
    """
    Download raw competition data for a given Kaggle competition.

    Args:
        taskname: The Kaggle competition slug.
        download_dir: Directory where the raw data will be stored.
    """
    os.makedirs(download_dir, exist_ok=True)
    input(
        f"Consent to the competition at "
        f"https://www.kaggle.com/competitions/{taskname}/data; "
        "Press any key after you have accepted the rules online."
    )
    # download and unzip
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", taskname],
        cwd=download_dir,
        check=True
    )
    subprocess.run(
        ["unzip", "-n", f"{taskname}.zip"],
        cwd=download_dir,
        check=True
    )
    os.remove(os.path.join(download_dir, f"{taskname}.zip"))

def split_train_data(download_dir: str = "./polymer"):
    """
    Split train.csv into train/valid/test sets with ratio 0.7/0.1/0.2
    and remove unnecessary files.
    
    Args:
        download_dir: Directory containing the downloaded data.
    """
    train_path = os.path.join(download_dir, "train.csv")
    
    if not os.path.exists(train_path):
        print(f"train.csv not found in {download_dir}")
        return
    
    # Load the training data
    print("Loading train.csv...")
    df = pd.read_csv(train_path)
    print(f"Original training data shape: {df.shape}")
    
    # First split: 70% train, 30% temp (which will be split into 10% valid, 20% test)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # Second split: from the 30%, create 10% valid (1/3 of temp) and 20% test (2/3 of temp)
    valid_df, test_df = train_test_split(temp_df, test_size=0.667, random_state=42)
    
    # Save the split datasets
    train_df.to_csv(os.path.join(download_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(download_dir, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(download_dir, "test.csv"), index=False)
    
    print(f"Train set shape: {train_df.shape} (70%)")
    print(f"Valid set shape: {valid_df.shape} (10%)")
    print(f"Test set shape: {test_df.shape} (20%)")
    
    # Remove sample submission file
    sample_submission_path = os.path.join(download_dir, "sample_submission.csv")
    if os.path.exists(sample_submission_path):
        os.remove(sample_submission_path)
        print("Removed sample_submission.csv")

def main():
    # 1) Download raw competition data
    start_time = time()
    taskname = "neurips-open-polymer-prediction-2025"
    download_dir = "./polymer"
    download_raw_data(taskname, download_dir)
    print(f"Raw competition data downloaded in {time() - start_time:.2f} seconds")
    
    # 2) Split the training data and clean up files
    split_start = time()
    split_train_data(download_dir)
    print(f"Data splitting completed in {time() - split_start:.2f} seconds")

if __name__ == "__main__":
    main()