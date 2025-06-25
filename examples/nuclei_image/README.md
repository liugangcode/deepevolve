# nuclei_image

## Problem Description

### Overview

Identifying the cells' nuclei is the starting point for most analyses because most of the human body's 30 trillion cells contain a nucleus full of DNA, the genetic code that programs each cell. Identifying nuclei allows researchers to identify each individual cell in a sample, and by measuring how cells react to various treatments, the researcher can understand the underlying biological processes at work.

By participating, teams will work to automate the process of identifying nuclei, which will allow for more efficient drug testing, shortening the 10 years it takes for each new drug to come to market. 

### What will participants do?

Teams will create a computer model that can identify a range of nuclei across varied conditions. By observing patterns, asking questions, and building a model, participants will have a chance to push state-of-the-art technology farther.

### Evaluation

This competition is evaluated on the mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

\[
\mathrm{IoU}(A, B) = \frac{\lvert A \cap B\rvert}{\lvert A \cup B\rvert}.
\]

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95). In other words, at a threshold of 0.5, a predicted object is considered a “hit” if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value \( t \), a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

\[
\mathrm{Precision}(t) = \frac{\mathrm{TP}(t)}{\mathrm{TP}(t) + \mathrm{FP}(t) + \mathrm{FN}(t)}.
\]

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:

\[
\text{Average Precision} = \frac{1}{\lvert \text{thresholds}\rvert} \sum_{t} \frac{\mathrm{TP}(t)}{\mathrm{TP}(t) + \mathrm{FP}(t) + \mathrm{FN}(t)}.
\]

Lastly, the score returned by the competition metric is the mean of the individual average precisions of each image in the test dataset.

### Dataset Description

This dataset contains a large number of segmented nuclei images. The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (brightfield vs. fluorescence). The dataset is designed to challenge an algorithm's ability to generalize across these variations.

Each image is represented by an associated ImageId. Files belonging to an image are contained in a folder with this ImageId. Within this folder are two subfolders:

- **images** contains the image file.
- **masks** contains the segmented masks of each nucleus. This folder is only included in the training set. Each mask contains one nucleus. Masks are not allowed to overlap (no pixel belongs to two masks).

The second stage dataset will contain images from unseen experimental conditions. To deter hand labeling, it will also contain images that are ignored in scoring. The metric used to score this competition requires that your submissions are in run-length encoded format.

As with any human-annotated dataset, you may find various forms of errors in the data. You may manually correct errors you find in the training set. The dataset will not be updated/re-released unless it is determined that there are a large number of systematic errors. The masks of the stage 1 test set will be released with the release of the stage 2 test set.

### File Descriptions

- `/stage1_train/*` - training set images (images and annotated masks)
- `/stage1_test/*` - test set images (images and annotated masks)

- **Evaluation Metric**: mean average precision across intersection-over-union thresholds from 0.5 to 0.95
- **Interface File**: `deepevolve_interface.py`

## Initial Idea

### Nucleus Detection with UNet

The initial approach applies a U-Net segmentation network to identify nuclei in microscopy images. We resize raw images to 256×256 pixels, normalize them to zero mean and unit variance, and convert ground-truth masks into unique integer labels via connected-component analysis. The network is trained with the Adam optimizer over up to 100 epochs using a soft Dice loss, with early stopping triggered when the validation Dice coefficient stops improving. At inference, the model produces probability maps that are thresholded at 0.5, and connected components are extracted as individual nucleus predictions.

For further details, please refer to the [Kaggle Notebook](https://www.kaggle.com/code/cloudfall/pytorch-tutorials-on-dsb2018).