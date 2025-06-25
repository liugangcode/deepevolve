import os
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from skimage import io
from skimage.measure import label
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import time
import copy

import torch as t
from torch.utils import data
from torchvision import transforms as tsf
from torch import nn
import torch.nn.functional as F

# TTY detection for conditional printing
is_tty = sys.stdout.isatty()

def conditional_print(*args, **kwargs):
    """Print only if output is to a TTY"""
    if is_tty:
        print(*args, **kwargs)

@dataclass
class Config:
    """Configuration class containing hyperparameters and paths"""
    # Data paths
    base_dir: str = "data_cache/nuclei_image"
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    
    # Control flags
    reprocess_cache: bool = False
    
    # Model hyperparameters
    n_channels: int = 3
    n_classes: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 128
    num_epochs: int = 100
    image_size: Tuple[int, int] = (256, 256)
    
    # Training parameters
    num_workers: int = 4
    random_state: int = 42
    
    # Normalization parameters
    mean: List[float] = (0.5, 0.5, 0.5)
    std: List[float] = (0.5, 0.5, 0.5)
    
    # Device configuration
    device: str = "cuda" if t.cuda.is_available() else "cpu"


# Model classes
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = t.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = t.nn.functional.sigmoid(x)
        return x


class Dataset(data.Dataset):
    def __init__(self, data, source_transform, target_transform):
        self.datas = data
        self.s_transform = source_transform
        self.t_transform = target_transform
        
    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img']
        if isinstance(img, t.Tensor):
            img = img.numpy()
        mask = data['mask']
        if isinstance(mask, t.Tensor):
            mask = mask.numpy()
        
        # Ensure mask has the right shape for transforms
        if len(mask.shape) == 2:
            mask = mask[:, :, None]
        
        img = self.s_transform(img)
        mask = self.t_transform(mask)
        return img, mask
        
    def __len__(self):
        return len(self.datas)


def process_single_file(file_path: Path) -> dict:
    """Process a single file - unified for both train and test"""
    item = {}
    
    # Process images
    imgs = []
    images_dir = file_path / 'images'
    if not images_dir.exists():
        conditional_print(f"Warning: No images directory found in {file_path}")
        return None
        
    for image in images_dir.iterdir():
        img = io.imread(image)
        imgs.append(img)
    
    if len(imgs) == 0:
        conditional_print(f"Warning: No images found in {images_dir}")
        return None
        
    assert len(imgs) == 1, f"Expected 1 image, found {len(imgs)} in {images_dir}"
    img = imgs[0]
    
    # Remove alpha channel if present
    if len(img.shape) == 3 and img.shape[2] > 3:
        assert (img[:,:,3] != 255).sum() == 0
        img = img[:,:,:3]

    # Process masks - unified approach
    masks_dir = file_path / 'masks'
    if masks_dir.exists():
        mask_files = list(masks_dir.iterdir())
        if len(mask_files) > 0:
            masks = None
            for ii, mask_file in enumerate(mask_files):
                mask = io.imread(mask_file)
                assert (mask[(mask != 0)] == 255).all()
                if masks is None:
                    H, W = mask.shape
                    masks = np.zeros((len(mask_files), H, W))
                masks[ii] = mask
            
            # Verify masks don't overlap
            tmp_mask = masks.sum(0)
            assert (tmp_mask[tmp_mask != 0] == 255).all()
            
            # Create combined mask with unique IDs
            for ii, mask in enumerate(masks):
                masks[ii] = mask/255 * (ii+1)
            combined_mask = masks.sum(0)
            item['mask'] = combined_mask.astype(np.float32)
        else:
            # No mask files found, create empty mask
            H, W = img.shape[:2]
            item['mask'] = np.zeros((H, W), dtype=np.float32)
    else:
        # No masks directory, create empty mask
        H, W = img.shape[:2]
        item['mask'] = np.zeros((H, W), dtype=np.float32)
    
    item['name'] = file_path.name
    item['img'] = img
    return item


def process_image_data(file_path: str, n_workers: int = 4) -> List[dict]:
    """Process data using multiprocessing - unified for both train and test"""
    file_path = Path(file_path)
    files = sorted(list(file_path.iterdir()))
    
    # Use multiprocessing
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, files),
            total=len(files),
            desc=f"Processing data from {file_path.name}",
            disable=not is_tty
        ))
    
    # Filter out None results and convert to tensors
    datas = []
    for item in results:
        if item is not None:
            item['img'] = t.from_numpy(item['img'])
            item['mask'] = t.from_numpy(item['mask'])
            datas.append(item)
    
    return datas


def preprocess_data(config: Config) -> Tuple[List[dict], List[dict], List[dict]]:
    """Preprocess training, validation, and test data"""
    conditional_print("Starting data preprocessing...")
    
    # Check if cached data exists and reprocess_cache flag
    if config.train_path is not None and config.test_path is not None:
        train_cache_exists = os.path.exists(config.train_path)
        test_cache_exists = os.path.exists(config.test_path)
        
        if train_cache_exists and test_cache_exists and not config.reprocess_cache:
            conditional_print("Loading cached data...")
            train_data = t.load(config.train_path)
            test_data = t.load(config.test_path)
            
            # Split train_data into train and validation
            train_split, val_split = split_data(train_data, config)
            return train_split, val_split, test_data
    
    conditional_print("Processing data from source...")
    
    # Process training data
    train_images_dir = os.path.join(config.base_dir, 'stage1_train')
    train_data = process_image_data(train_images_dir, n_workers=config.num_workers)
    
    # Process test data - same way as training data
    test_images_dir = os.path.join(config.base_dir, 'stage1_test')
    test_data = process_image_data(test_images_dir, n_workers=config.num_workers)
    
    # Split training data
    train_split, val_split = split_data(train_data, config)
    
    conditional_print(f"Training samples: {len(train_split)}")
    conditional_print(f"Validation samples: {len(val_split)}")
    conditional_print(f"Test samples: {len(test_data)}")

    if config.train_path is not None:
        t.save(train_split, config.train_path)
    if config.test_path is not None:
        t.save(test_data, config.test_path)
    
    return train_split, val_split, test_data


def split_data(train_data: List[dict], config: Config) -> Tuple[List[dict], List[dict]]:
    """Split training data into training and validation sets with 0.8/0.2 ratio"""
    if len(train_data) == 0:
        return [], []
    
    train_indices, val_indices = train_test_split(
        range(len(train_data)), 
        test_size=0.2, 
        random_state=config.random_state
    )
    
    train_split = [train_data[i] for i in train_indices]
    val_split = [train_data[i] for i in val_indices]
    
    return train_split, val_split


def create_data_loaders(train_data: List[dict], val_data: List[dict], test_data: List[dict], config: Config) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """Create PyTorch data loaders for training, validation, and test"""
    # Define transforms
    s_trans = tsf.Compose([
        tsf.ToPILImage(),
        tsf.Resize(config.image_size),
        tsf.ToTensor(),
        tsf.Normalize(mean=config.mean, std=config.std)
    ])
    
    t_trans = tsf.Compose([
        tsf.ToPILImage(),
        tsf.Resize(config.image_size, interpolation=Image.NEAREST),
        tsf.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = Dataset(train_data, s_trans, t_trans)
    val_dataset = Dataset(val_data, s_trans, t_trans)
    test_dataset = Dataset(test_data, s_trans, t_trans)
    
    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset, 
        num_workers=config.num_workers, 
        batch_size=config.batch_size,
        shuffle=True
    )
    
    val_loader = data.DataLoader(
        val_dataset, 
        num_workers=config.num_workers, 
        batch_size=config.batch_size,
        shuffle=False
    )
    
    test_loader = data.DataLoader(
        test_dataset, 
        num_workers=config.num_workers, 
        batch_size=config.batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def extract_objects(mask, min_size=10, is_prediction=False):
    """Extract individual objects from mask"""
    if mask.max() <= 0:
        return []
    
    if is_prediction:
        # For model predictions: convert continuous values to binary, then find connected components
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        if binary_mask.sum() == 0:
            return []
        
        # Label connected components to get separate objects
        labeled_mask = label(binary_mask)
        
        # Extract each connected component as separate object
        objects = []
        for region_id in range(1, labeled_mask.max() + 1):
            object_mask = (labeled_mask == region_id).astype(np.uint8)
            if object_mask.sum() >= min_size:
                objects.append(object_mask)
        
        return objects
    else:
        # For ground truth: extract objects by unique integer IDs
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background
        
        objects = []
        for obj_id in unique_ids:
            # Extract individual nucleus
            object_mask = (mask == obj_id).astype(np.uint8)
            
            # Check size threshold
            if object_mask.sum() >= min_size:
                objects.append(object_mask)
        
        return objects


def calculate_iou_vectorized(pred_objects, true_objects):
    """Vectorized IoU calculation for multiple object pairs"""
    if len(pred_objects) == 0 or len(true_objects) == 0:
        return np.zeros((len(pred_objects), len(true_objects)))
    
    # Stack masks for vectorized operations
    pred_stack = np.stack(pred_objects)  # Shape: (n_pred, H, W)
    true_stack = np.stack(true_objects)  # Shape: (n_true, H, W)
    
    # Reshape for broadcasting
    pred_expanded = pred_stack[:, None, :, :]  # Shape: (n_pred, 1, H, W)
    true_expanded = true_stack[None, :, :, :]  # Shape: (1, n_true, H, W)
    
    # Vectorized intersection and union
    intersection = np.logical_and(pred_expanded, true_expanded).sum(axis=(2, 3))
    union = np.logical_or(pred_expanded, true_expanded).sum(axis=(2, 3))
    
    # Avoid division by zero
    iou_matrix = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    
    return iou_matrix


def calculate_average_precision(pred_mask, true_mask, thresholds=None):
    """Calculate average precision with optimized matching"""
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.05)
    
    # Extract objects with appropriate method for each mask type
    pred_objects = extract_objects(pred_mask, is_prediction=True)
    true_objects = extract_objects(true_mask, is_prediction=False)
    
    # Early return for edge cases
    if len(pred_objects) == 0 and len(true_objects) == 0:
        return 1.0
    
    if len(pred_objects) == 0 or len(true_objects) == 0:
        return 0.0
    
    # Calculate IoU matrix once for all thresholds
    iou_matrix = calculate_iou_vectorized(pred_objects, true_objects)
    
    # Calculate precision at each threshold using the same IoU matrix
    precisions = []
    for threshold in thresholds:
        # Use precomputed IoU matrix
        valid_matches = iou_matrix > threshold
        
        if not valid_matches.any():
            precision = 0.0
        else:
            # Efficient matching using greedy approach
            matched_pred = set()
            matched_true = set()
            
            pred_idx, true_idx = np.where(valid_matches)
            iou_values = iou_matrix[pred_idx, true_idx]
            sort_indices = np.argsort(-iou_values)
            
            for idx in sort_indices:
                p_idx, t_idx = pred_idx[idx], true_idx[idx]
                if p_idx not in matched_pred and t_idx not in matched_true:
                    matched_pred.add(p_idx)
                    matched_true.add(t_idx)
            
            true_positives = len(matched_pred)
            false_positives = len(pred_objects) - true_positives
            false_negatives = len(true_objects) - len(matched_true)
            
            denominator = true_positives + false_positives + false_negatives
            precision = true_positives / denominator if denominator > 0 else 1.0
        
        precisions.append(precision)
    
    return np.mean(precisions)


def evaluate_dataset_map(model: nn.Module, data_loader: data.DataLoader, device: t.device, dataset_name: str = "Dataset") -> float:
    """Evaluate mAP on dataset"""
    conditional_print(f"Evaluating {dataset_name} using mAP metric...")
    
    model = model.to(device)
    model.eval()
    
    all_average_precisions = []
    thresholds = np.arange(0.5, 1.0, 0.05)
    
    with t.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(data_loader, desc=f"Evaluating {dataset_name}", disable=not is_tty)):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Process batch
            for i in range(images.size(0)):
                pred_mask = outputs[i][0].cpu().numpy()
                true_mask = masks[i][0].cpu().numpy()
                
                # Calculate AP for this image
                ap = calculate_average_precision(pred_mask, true_mask, thresholds)
                all_average_precisions.append(ap)
            
    if len(all_average_precisions) == 0:
        conditional_print(f"No valid data for {dataset_name} evaluation")
        return 0.0
    
    mean_ap = np.mean(all_average_precisions)
    conditional_print(f"{dataset_name} mAP: {mean_ap:.4f}")
    
    return mean_ap


def soft_dice_loss(inputs: t.Tensor, targets: t.Tensor) -> t.Tensor:
    """Calculate Soft Dice Loss"""
    num = targets.size(0)
    m1 = inputs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    score = 1 - score.sum() / num
    return score


def dice_coefficient(inputs: t.Tensor, targets: t.Tensor) -> float:
    """Calculate Dice coefficient for evaluation"""
    num = targets.size(0)
    m1 = inputs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    return score.mean().item()


def train_model(model: nn.Module, train_loader: data.DataLoader, val_loader: data.DataLoader, config: Config, device: t.device) -> nn.Module:
    """Train the UNet model with early stopping"""
    conditional_print(f"Starting model training on device: {device}")
    
    # Move model to device
    model = model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Early stopping parameters
    best_val_dice = 0.0
    patience = 50
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        for x_train, y_train in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} - Training", disable=not is_tty):
            # Move data to device
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = soft_dice_loss(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches
        
        # Validation phase
        model.eval()
        total_val_dice = 0
        num_val_batches = 0
        
        with t.no_grad():
            for x_val, y_val in val_loader:
                # Move data to device
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                
                outputs = model(x_val)
                dice = dice_coefficient(outputs, y_val)
                total_val_dice += dice
                num_val_batches += 1
        
        avg_val_dice = total_val_dice / num_val_batches if num_val_batches > 0 else 0
        
        # Early stopping check
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            patience_counter = 0
            # Save best model state
            best_model_state = copy.deepcopy(model.state_dict())
            conditional_print(f"Epoch {epoch+1}/{config.num_epochs} - New best validation Dice: {best_val_dice:.4f}")
        else:
            patience_counter += 1
        
        conditional_print(f"Epoch {epoch+1}/{config.num_epochs}")
        conditional_print(f"  Train Loss: {avg_train_loss:.4f}")
        conditional_print(f"  Val Dice: {avg_val_dice:.4f}")
        conditional_print(f"  Best Val Dice: {best_val_dice:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            conditional_print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        conditional_print(f"Restored model to best state with validation Dice: {best_val_dice:.4f}")
    
    return model


def main(config: Config):
    """Main function to run the complete pipeline"""
    
    config.train_path = f"{config.base_dir}/train.pth"
    config.test_path = f"{config.base_dir}/test.pth"

    # Set up device
    device = t.device(config.device)
    conditional_print(f"Using device: {device}")
    
    # Step 1: Preprocess data
    train_split, val_split, test_data = preprocess_data(config)
    
    if len(train_split) == 0:
        conditional_print("No training data found. Please check the data path.")
        raise Exception("No training data found")
    
    # Step 2: Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_split, val_split, test_data, config)
    
    # Step 3: Initialize and train model
    model = UNet(config.n_channels, config.n_classes)
    trained_model = train_model(model, train_loader, val_loader, config, device)
    
    # Step 4: Evaluate model using mAP metric
    conditional_print("\n" + "="*60)
    conditional_print("FINAL EVALUATION USING mAP METRIC")
    conditional_print("="*60)
    
    # Evaluate on training set (sample for speed)
    conditional_print("Evaluating on training set (sampling for speed)...")
    train_subset = data.Subset(train_loader.dataset, list(range(0, len(train_loader.dataset), 5)))
    train_subset_loader = data.DataLoader(train_subset, batch_size=config.batch_size, shuffle=False)
    train_map = evaluate_dataset_map(trained_model, train_subset_loader, device, "Training (sampled)")
    
    # Evaluate on validation set
    valid_map = evaluate_dataset_map(trained_model, val_loader, device, "Validation")

    # Evaluate on test set
    test_map = evaluate_dataset_map(trained_model, test_loader, device, "Test")

    # Print final results
    conditional_print("\n" + "="*60)
    conditional_print("FINAL RESULTS")
    conditional_print("="*60)
    conditional_print(f"Training mAP (sampled): {train_map:.4f}")
    conditional_print(f"Validation mAP:         {valid_map:.4f}")
    conditional_print(f"Test mAP:               {test_map:.4f}")
    conditional_print("="*60)
    
    results = {
        "train_map": train_map,
        "valid_map": valid_map,
        "test_map": test_map
    }
    
    return results


if __name__ == "__main__":
    config = Config()
    config.base_dir = "../../../data_cache/nuclei_image"
    # config.num_epochs = 10
    results = main(config)
    print(results)