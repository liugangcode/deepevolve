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

def download_solution_data(dataset_list, target_dir: str = "./openvaccine/solution"):
    """
    Download and unzip multiple Kaggle datasets into target_dir.
    Each dataset goes into its own subfolder named after the dataset.
    CSV.gz files are not unzipped.
    Skips downloading if the solution folder already exists.

    Args:
        dataset_list: List of Kaggle dataset slugs.
        target_dir: Directory where the solution data will be stored.
    """
    os.makedirs(target_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    for ds in dataset_list:
        # Extract solution name from dataset slug
        solution_name = ds.split('/')[-1]  # Get the part after the last '/'
        solution_dir = os.path.join(target_dir, solution_name)
        
        # Check if the solution folder already exists
        if os.path.exists(solution_dir) and os.listdir(solution_dir):
            print(f"Skipping {ds} - folder already exists: {solution_dir}")
            continue
        
        print(f"Downloading {ds} …")
        os.makedirs(solution_dir, exist_ok=True)
        
        # Download the dataset
        api.dataset_download_files(
            ds,
            path=solution_dir,
            unzip=True,
            quiet=False
        )
    
    print("All solution datasets downloaded.")

def main():
    # 1) Download raw competition data
    start_time = time()
    taskname = "stanford-covid-vaccine"
    download_dir = "./openvaccine"
    download_raw_data(taskname, download_dir)
    print(f"Raw competition data downloaded in {time() - start_time:.2f} seconds")

    # start_time = time()
    # solution_datasets = [
    #     "kfujikawa/stanford-covid-vaccine-new-sequences-augmentation",
    #     "onodera/stanford-covid-vaccine-onodera-models",
    #     "onodera/covid-233-kf-outputs",
    #     "onodera/covid-233-onodera-outputs-v2"
    # ]
    # solution_dir = os.path.join(download_dir, "solution")
    # download_solution_data(solution_datasets, solution_dir)
    # print(f"Solution datasets downloaded in {time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
