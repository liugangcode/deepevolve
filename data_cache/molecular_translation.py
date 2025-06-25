import os
import subprocess
import zipfile
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

# Silence non-critical RDKit InChI warnings (for example "Mobile-H" messages)
RDLogger.DisableLog("rdkit.Inchi")

# --- configuration ---
TASKNAME    = "bms-molecular-translation"
BASE_DIR    = "./molecular_translation"
ZIP_PATH    = os.path.join(BASE_DIR, f"{TASKNAME}.zip")
LABELS_CSV  = os.path.join(BASE_DIR, "train_labels.csv")
IMAGE_DIR   = os.path.join(BASE_DIR, "images")
MAX_SIZE_MB = 100
SAMPLE_SIZE = 50_000

def inchi_to_smiles_safe(inchi: str):
    """Return a canonical SMILES string, or None if conversion fails."""
    try:
        mol = Chem.MolFromInchi(inchi, sanitize=True, removeHs=False)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None

def add_smiles_column(csv_path: Path, num_workers: int = 6) -> None:
    """Convert InChI to SMILES in parallel, save results, and log failures."""
    df = pd.read_csv(csv_path, dtype={"image_id": str, "InChI": str})
    if "InChI" not in df.columns:
        raise ValueError(f"`InChI` column not found in {csv_path}")

    inchi_list = df["InChI"].tolist()

    # Parallel conversion with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        smiles_list = list(
            tqdm(
                executor.map(inchi_to_smiles_safe, inchi_list),
                total=len(inchi_list),
                desc=f"Processing {csv_path.name}"
            )
        )

    df["SMILES"] = smiles_list
    df.to_csv(csv_path, index=False)

    # Identify failures
    failures = [(inchi, idx) for idx, (inchi, smi) in enumerate(zip(inchi_list, smiles_list)) if smi is None]
    n_failed = len(failures)

    if n_failed:
        # Save failures to a side-car CSV for later inspection
        fail_df = pd.DataFrame(failures, columns=["InChI", "row_index"])
        fail_path = csv_path.with_suffix(".failed.csv")
        fail_df.to_csv(fail_path, index=False)

    # Report summary
    print(f"{csv_path.name}: {n_failed} failures")
    if n_failed:
        print("Failed InChI strings:")
        for inchi, _ in failures:
            print("  ", inchi)

def download_and_unpack_labels():
    input(
        f"Consent to the competition at "
        f"https://www.kaggle.com/competitions/{TASKNAME}/data; "
        "Press any key after you have accepted the rules online."
    )
    os.makedirs(BASE_DIR, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", TASKNAME, "-p", BASE_DIR],
        check=True
    )
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extract("train_labels.csv", BASE_DIR)

def safe_move(src, dst):
    """Safely move a file, handling AFS filesystem limitations."""
    try:
        # Try normal move first
        shutil.move(src, dst)
    except OSError as e:
        if e.errno == 27:  # File too large error on AFS
            # Fall back to copy and delete
            try:
                shutil.copy2(src, dst)
                os.remove(src)
            except OSError:
                # If copy2 fails due to metadata issues, use basic copy
                try:
                    shutil.copy(src, dst)
                    os.remove(src)
                except OSError as copy_error:
                    # If all copy methods fail, print warning and continue
                    print(f"Warning: Failed to move {src} to {dst}: {copy_error}")
                    print(f"Skipping file and continuing...")
                    return False
        else:
            # For other OSErrors, print warning and continue
            print(f"Warning: Failed to move {src} to {dst}: {e}")
            print(f"Skipping file and continuing...")
            return False
    return True

def subsample_and_extract():
    # load labels and pick a random subset
    df = pd.read_csv(LABELS_CSV)
    subsample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    ids = subsample['image_id'].tolist()

    # open zip once
    print(f"Extracting {len(ids)} images from {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        members = []
        success_ids = []
        for img_id in ids:
            a, b, c = img_id[0], img_id[1], img_id[2]
            member = f"train/{a}/{b}/{c}/{img_id}.png"
            try:
                info = z.getinfo(member)
                if info.file_size <= MAX_SIZE_MB * 1024 * 1024:
                    members.append(member)
                    success_ids.append(img_id)
            except KeyError:
                print(f"Missing in ZIP: {member}")

        os.makedirs(IMAGE_DIR, exist_ok=True)
        # extract all selected files at once
        z.extractall(path=IMAGE_DIR, members=members)

    # flatten using system mv command in parallel batches
    print(f"Flattening {len(success_ids)} images using system commands...")
    
    def move_batch(batch):
        """Move a batch of files using system mv command."""
        moved_ids = []
        for img_id in batch:
            a, b, c = img_id[0], img_id[1], img_id[2]
            src = os.path.join(IMAGE_DIR, 'train', a, b, c, f"{img_id}.png")
            dst = os.path.join(IMAGE_DIR, f"{img_id}.png")
            
            try:
                # Use system mv command which is usually faster
                result = subprocess.run(['mv', src, dst], capture_output=True)
                if result.returncode == 0:
                    moved_ids.append(img_id)
                else:
                    print(f"Warning: Failed to move {src}: {result.stderr.decode()}")
            except Exception as e:
                print(f"Warning: Failed to move {src}: {e}")
        
        return moved_ids
    
    # Split into batches for parallel processing
    batch_size = 500
    batches = [success_ids[i:i+batch_size] for i in range(0, len(success_ids), batch_size)]
    
    actually_moved = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_batch = {executor.submit(move_batch, batch): batch for batch in batches}
        
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_results = future.result()
            actually_moved.extend(batch_results)
            print(f"Progress: {len(actually_moved)}/{len(success_ids)} files moved")
    
    # remove the now-empty nested structure
    print(f"Removing nested structure...")
    shutil.rmtree(os.path.join(IMAGE_DIR, 'train'))

    # report any failures
    failures = set(ids) - set(actually_moved)
    for img_id in failures:
        print(f"Failed to extract or move {img_id}")

    return subsample[subsample['image_id'].isin(actually_moved)]

def split_and_save(df):
    # 60/20/20 train/valid/test split
    train, temp = train_test_split(df, train_size=0.6, random_state=42)
    valid, test = train_test_split(temp, train_size=0.5, random_state=42)
    for name, split_df in [('train', train), ('valid', valid), ('test', test)]:
        path = os.path.join(BASE_DIR, f"{name}.csv")
        split_df.to_csv(path, index=False)
        print(f"Saved {path} ({len(split_df)} rows)")

def convert_inchi_to_smiles():
    """Convert InChI to SMILES for all CSV files."""
    print("\nConverting InChI to SMILES...")
    for filename in ("train.csv", "valid.csv", "test.csv"):
        csv_path = Path(BASE_DIR) / filename
        if csv_path.exists():
            add_smiles_column(csv_path, num_workers=6)
        else:
            print(f"Warning: {csv_path} not found, skipping SMILES conversion")

def main():
    if not os.path.exists(ZIP_PATH):
        download_and_unpack_labels()
    else:
        print(f"ZIP file {ZIP_PATH} already exists, skipping download")

    subsample_df = subsample_and_extract()
    split_and_save(subsample_df)
    
    # Convert InChI to SMILES for all split files
    # convert_inchi_to_smiles()

    # remove zip to save space
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    
    if os.path.exists(LABELS_CSV):
        os.remove(LABELS_CSV)

    print("Done: CSV files in BASE_DIR and images in IMAGE_DIR.")

if __name__ == "__main__":
    main()