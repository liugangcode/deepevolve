# Nuclei Image Segmentation

## Overview

Identifying the cells' nuclei is the starting point for most analyses because most of the human body's 30 trillion cells contain a nucleus full of DNA, the genetic code that programs each cell. By isolating nuclei, researchers can identify individual cells within a sample and analyze how they respond to various treatments. This is crucial for understanding the underlying biological processes at work.

By participating in this challenge, teams will work on automating the process of nuclei identification, which has the potential to significantly accelerate drug testing, thereby reducing the time required to bring new drugs to market.

## Evaluation

The competition is evaluated on the mean average precision across a range of intersection over union (IoU) thresholds. For any two sets, the IoU is defined as:

```math
\mathrm{IoU}(A, B) = \frac{\lvert A \cap B\rvert}{\lvert A \cup B\rvert}.
```

The competition metric sweeps over threshold values from 0.5 to 0.95 with a step size of 0.05 (i.e., 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95). At each threshold \(t\), precision is calculated as:

```math
\mathrm{Precision}(t) = \frac{\mathrm{TP}(t)}{\mathrm{TP}(t) + \mathrm{FP}(t) + \mathrm{FN}(t)},
```

where:
- A true positive (TP) is counted when a predicted object matches a ground truth object with an IoU above the threshold.
- A false positive (FP) indicates a predicted object that has no associated ground truth object.
- A false negative (FN) indicates a ground truth object that has no associated predicted object.

The average precision for a single image is then calculated as:

```math
\text{Average Precision} = \frac{1}{\lvert \text{thresholds} \rvert} \sum_{t} \frac{\mathrm{TP}(t)}{\mathrm{TP}(t) + \mathrm{FP}(t) + \mathrm{FN}(t)}.
```

Finally, the competition metric score is the mean of the individual average precisions computed for each image in the test dataset.

## Dataset Description

This dataset comprises a large number of segmented nuclei images acquired under various conditions. Images vary by cell type, magnification, and imaging modality (brightfield vs. fluorescence), posing a significant challenge for algorithms to generalize across different conditions.

Each image is represented by an associated `ImageId`. Files for a given image reside in a folder named after its `ImageId`. Within each folder, there are two subfolders:

- **images**: Contains the raw image file.
- **masks**: Contains the segmented masks, where each mask delineates a single nucleus. Note that:
  - Masks in the training set are provided, and each nucleus is given a unique integer label through connected-components analysis.
  - Masks do not overlap; no pixel belongs to two masks.

For the second stage, the dataset will include images from unseen experimental conditions. Some images in this stage will be ignored in scoring to prevent hand labeling, and submissions are required to be in run-length encoded format. Please refer to the evaluation page for submission details.

Data directories overview:
- `/stage1_train/*` - Training set images (includes images and annotated masks)
- `/stage1_test/*` - Test set images (images and annotated masks)

## Interface

The project leverages the provided `deepevolve_interface.py` to interact with the dataset and evaluation process.

## Initial Approach: Nucleus Detection with U-Net

### Methodology

The initial strategy employs a U-Net segmentation network to identify nuclei in microscopy images:

- **Preprocessing**:  
  - Resize raw images to 256×256 pixels.
  - Normalize images to have zero mean and unit variance.
  - Convert ground-truth masks into unique integer labels using connected-component analysis.

- **Training**:  
  - Use the Adam optimizer and train the network for up to 100 epochs.
  - Apply a soft Dice loss function.
  - Implement early stopping based on the validation Dice coefficient.

- **Inference**:  
  - The trained model outputs probability maps.
  - Threshold the probability maps at 0.5.
  - Extract connected components from the thresholded maps to obtain individual nucleus predictions.