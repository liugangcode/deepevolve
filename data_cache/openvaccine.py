import os
import subprocess
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from time import time

def download_raw_data(taskname: str, download_dir: str = "./openvaccine"):
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

def main():
    # 1) Download raw competition data
    start_time = time()
    taskname = "stanford-covid-vaccine"
    download_dir = "./openvaccine"
    download_raw_data(taskname, download_dir)
    print(f"Raw competition data downloaded in {time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
