from __future__ import annotations
import os
import argparse
from pathlib import Path

import pandas as pd
from torchvision.datasets.utils import download_url
from tqdm import tqdm
import h5py
import numpy as np

# size info: https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download

def parse_metadata(pde_name: str) -> pd.DataFrame:
    """
    Read the CSV of URLs and filter to the given PDE.
    """
    csv_path = Path(__file__).with_name("pdebench_data_urls.csv")
    meta_df = pd.read_csv(csv_path)
    meta_df["PDE"] = meta_df["PDE"].str.lower()

    valid = {
        "advection", "burgers", "1d_cfd", "diff_sorp", "1d_reacdiff",
        "2d_cfd", "darcy", "2d_reacdiff", "ns_incom", "swe", "3d_cfd",
    }
    pde = pde_name.lower()
    assert pde in valid, f"PDE name '{pde_name}' not recognized."

    return meta_df[meta_df["PDE"] == pde]

def download_data(pde_name: str):
    """
    Download all HDF5 files for a given PDE into root_folder/<Path> directories.
    """
    pde_df = parse_metadata(pde_name)
    target_dir = Path(pde_name) / "original"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if all files already exist
    all_files_exist = True
    for _, row in pde_df.iterrows():
        file_path = target_dir / row["Filename"]
        if not file_path.exists():
            all_files_exist = False
            break
    
    if all_files_exist:
        print(f"All files for '{pde_name}' already exist. Skipping download.")
        return
    
    print(f"Downloading missing files for '{pde_name}'...")
    for _, row in tqdm(pde_df.iterrows(), total=len(pde_df), desc="Downloading"):
        file_path = target_dir / row["Filename"]
        if file_path.exists():
            print(f"File {row['Filename']} already exists. Skipping.")
            continue
        download_url(row["URL"], str(target_dir), row["Filename"], md5=row["MD5"])

def work(dataset_path, subset_path, subset_selection):
    # Skip if subset file already exists
    if os.path.exists(subset_path):
        print(f"Subset file {subset_path} already exists. Skipping.")
        return
    
    # Load data from file
    with h5py.File(dataset_path, 'r') as f:
        # Load the data
        print(f"Available keys in {dataset_path}: {list(f.keys())}")
        t_coordinate = np.array(f['t-coordinate'])[:-1]  # Keep as is
        x_coordinate = np.array(f['x-coordinate'])  # Keep as is
        u = subset_selection(np.array(f['tensor']))

        # Navier-Stokes data has different structure
        # Vx = subset_selection((f['Vx']))
        # density = subset_selection(np.array(f['density']))
        # pressure = subset_selection(np.array(f['pressure']))

    # Verify shapes
    print(t_coordinate.shape, x_coordinate.shape, u.shape)
    # (201,) (1024,) (100, 201, 1024) for burgers equation

    # Save the subset to a new HDF5 file
    with h5py.File(subset_path, 'w') as f:
        # Create datasets in the new file
        f.create_dataset('t-coordinate', data=t_coordinate)
        f.create_dataset('tensor', data=u)
        f.create_dataset('x-coordinate', data=x_coordinate)

        # Uncomment if you want to save Navier-Stokes specific data
        # f.create_dataset('Vx', data=Vx)
        # f.create_dataset('density', data=density)
        # f.create_dataset('pressure', data=pressure)

    print(f"Subset data saved successfully at {subset_path}!")

if __name__ == '__main__':
    pde_name = 'burgers'

    test_subset_size = 100
    dev_subset_size = 50

    download_data(pde_name)

    dataset_dir = Path(pde_name) / "original"
    for item in os.listdir(dataset_dir):
        full_path = os.path.join(dataset_dir, item)
        if os.path.isfile(full_path):
            print(full_path)

            subset_path = os.path.join(pde_name, item)
            work(full_path, subset_path, lambda x: x[:test_subset_size])

            development_subset_path = subset_path.replace('.hdf5', '_development.hdf5')
            work(full_path, development_subset_path, lambda x: x[-dev_subset_size:])

    print(f"Done. Subsets are in ./{pde_name}/")