# from: https://github.com/snap-stanford/MLAgentBench/blob/main/MLAgentBench/benchmarks/amp-parkinsons-disease-progression-prediction/scripts/prepare.py

import subprocess
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split

taskname = "us-patent-phrase-to-phrase-matching"
download_dir = "./usp_p2p"
os.makedirs(download_dir, exist_ok=True)

input(f"Consent to the competition at https://www.kaggle.com/competitions/{taskname}/data; Press any key after you have accepted the rules online.")

subprocess.run(["kaggle", "competitions", "download", "-c", taskname], cwd=download_dir) 
subprocess.run(["unzip", "-n", f"{taskname}.zip"], cwd=download_dir) 
subprocess.run(["rm", f"{taskname}.zip"], cwd=download_dir) 

data_dir = os.path.dirname(__file__)
train_path = os.path.join(data_dir, download_dir, "train.csv")
test_path = os.path.join(data_dir, download_dir, "test.csv")
sample_submission_path = os.path.join(data_dir, download_dir, "sample_submission.csv")

# 1. Remove the current test.csv if it exists
if os.path.exists(test_path):
    os.remove(test_path)
    print(f"Removed existing {test_path}")

# 2. Read train.csv
df = pd.read_csv(train_path)

# 3. Split into 90% train, 10% test
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# 4. Save the new splits
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"Split complete. New train: {len(train_df)} rows, new test: {len(test_df)} rows.")

# 5. Create sample_submission.csv with id from test.csv and score=0
sample_submission = pd.DataFrame({
    "id": test_df["id"],
    "score": 0
})
sample_submission.to_csv(sample_submission_path, index=False)
print(f"Created {sample_submission_path} with {len(sample_submission)} rows.")