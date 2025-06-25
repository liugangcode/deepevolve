import os
import zipfile
import shutil
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import subprocess
import pandas as pd

DEBUG = False  # Set True for debugging with one image

def download_raw_data(taskname: str, download_dir: str):
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


# ------------------------------------------------------------------
# RLE helpers
# ------------------------------------------------------------------
def rle_decode(rle_str: str, mask_shape, mask_dtype=np.uint8) -> np.ndarray:
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T

def rle_encode(mask: np.ndarray) -> np.ndarray:
    pixels = mask.T.flatten()
    pad = pixels[0] or pixels[-1]
    if pad:
        pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if pad:
        runs -= 1
    runs[1::2] -= runs[:-1:2]
    return runs

def rle_to_string(runs: np.ndarray) -> str:
    return ' '.join(str(x) for x in runs)


def human_readable_size(size_bytes: int) -> str:
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024 or unit == 'TB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024

def unzip_archives(directory: str):
    for fname in os.listdir(directory):
        if not fname.lower().endswith('.zip'):
            continue
        zip_path = os.path.join(directory, fname)
        base_name = os.path.splitext(fname)[0]
        with zipfile.ZipFile(zip_path, 'r') as z:
            file_entries = [zi for zi in z.infolist() if not zi.is_dir()]
            if len(file_entries) == 1:
                print(f"Extracting single file from {fname}")
                z.extractall(path=directory)
            else:
                target_dir = os.path.join(directory, base_name)
                os.makedirs(target_dir, exist_ok=True)
                print(f"Extracting {len(file_entries)} files from {fname} into {target_dir}")
                z.extractall(path=target_dir)
        os.remove(zip_path)

def remove_stage2(directory: str):
    for entry in os.listdir(directory):
        if entry.startswith('stage2_'):
            path = os.path.join(directory, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            else:
                os.remove(path)
                print(f"Removed file: {path}")

# ------------------------------------------------------------------
# Decode and verify masks
# ------------------------------------------------------------------
def decode_solution_to_masks(download_dir: str):
    solution_csv = Path(download_dir) / 'stage1_solution.csv'
    stage1_test = Path(download_dir) / 'stage1_test'

    df = pd.read_csv(solution_csv)
    for idx, (image_id, group) in enumerate(df.groupby("ImageId")):
        height, width = int(group.iloc[0]["Height"]), int(group.iloc[0]["Width"])
        masks_dir = stage1_test / image_id / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        for k, row in group.reset_index(drop=True).iterrows():
            mask = rle_decode(row["EncodedPixels"], (height, width))
            mask_path = masks_dir / f"{image_id}_mask_{k}.png"
            imageio.imwrite(mask_path.as_posix(), (mask * 255).astype(np.uint8))

        print(f"Decoded masks saved for {image_id}")
        if DEBUG:
            break

def verify_train_rle_encoding(download_dir: str):
    train_csv = Path(download_dir) / 'stage1_train_labels.csv'
    stage1_train = Path(download_dir) / 'stage1_train'

    df = pd.read_csv(train_csv)
    mismatches = 0

    for idx, (image_id, group) in enumerate(df.groupby("ImageId")):
        mask_dir = stage1_train / image_id / "masks"
        rle_file_set = set(group["EncodedPixels"])
        rle_new_set = set()

        for m_path in sorted(mask_dir.glob("*.png")):
            mask = imageio.imread(m_path.as_posix()) > 0
            rle_new_set.add(rle_to_string(rle_encode(mask.astype(np.uint8))))

        if rle_file_set != rle_new_set:
            print(f"RLE mismatch in {image_id}")
            mismatches += 1

        if DEBUG:
            break

    print(f"Verification finished; mismatched images: {mismatches}")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    download_dir = "./nuclei_image2"

    download_raw_data("data-science-bowl-2018", download_dir)

    start = time()
    unzip_archives(download_dir)
    print(f"\nAll archives processed in {time() - start:.2f} seconds\n")

    remove_stage2(download_dir)

    decode_solution_to_masks(download_dir)
    verify_train_rle_encoding(download_dir)

if __name__ == "__main__":
    main()


# import os
# import zipfile
# from time import time
# import shutil

# def estimate_unzip_size(directory: str) -> int:
#     """
#     Sum the uncompressed sizes of all .zip files in `directory`.
#     Returns total size in bytes.
#     """
#     total = 0
#     for fname in os.listdir(directory):
#         if fname.lower().endswith('.zip'):
#             path = os.path.join(directory, fname)
#             with zipfile.ZipFile(path, 'r') as z:
#                 for zi in z.infolist():
#                     # skip directory entries
#                     if not zi.is_dir():
#                         total += zi.file_size
#     return total

# def human_readable_size(size_bytes: int) -> str:
#     """
#     Convert a size in bytes to a human-readable string.
#     """
#     for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
#         if size_bytes < 1024 or unit == 'TB':
#             return f"{size_bytes:.2f} {unit}"
#         size_bytes /= 1024

# def unzip_archives(directory: str):
#     """
#     For each .zip in `directory`:
#       - If it has exactly one file entry, extract that file into `directory`.
#       - Otherwise, make a subfolder (named like the zip, without .zip)
#         and extract all contents there.
#       Then remove the .zip file.
#     """
#     for fname in os.listdir(directory):
#         if not fname.lower().endswith('.zip'):
#             continue
#         zip_path = os.path.join(directory, fname)
#         base_name = os.path.splitext(fname)[0]
#         with zipfile.ZipFile(zip_path, 'r') as z:
#             # count non-directory entries
#             file_entries = [zi for zi in z.infolist() if not zi.is_dir()]
#             if len(file_entries) == 1:
#                 # single file: extract directly
#                 print(f"Extracting single file from {fname}")
#                 z.extractall(path=directory)
#             else:
#                 # multiple files: extract into subfolder
#                 target_dir = os.path.join(directory, base_name)
#                 os.makedirs(target_dir, exist_ok=True)
#                 print(f"Extracting {len(file_entries)} files from {fname} into {target_dir}")
#                 z.extractall(path=target_dir)
#         # remove the zip to save space
#         os.remove(zip_path)

# def remove_stage2(directory: str):
#     """
#     Remove any file or folder in `directory` whose name starts with 'stage2_'.
#     """
#     for entry in os.listdir(directory):
#         if entry.startswith('stage2_'):
#             path = os.path.join(directory, entry)
#             if os.path.isdir(path):
#                 shutil.rmtree(path)
#                 print(f"Removed directory: {path}")
#             else:
#                 os.remove(path)
#                 print(f"Removed file: {path}")


# def main():
#     download_dir = "./nuclei_image"
#     est_bytes = estimate_unzip_size(download_dir)
#     print(f"Estimated total size after unzip: {est_bytes} bytes ({human_readable_size(est_bytes)})\n")

#     start = time()
#     unzip_archives(download_dir)
#     print(f"\nAll archives processed in {time() - start:.2f} seconds")

#     remove_stage2(download_dir)

# if __name__ == "__main__":
#     main()
