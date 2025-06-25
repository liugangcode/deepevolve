# nuclei_image

## Problem Description

### Overview

Identifying cell nuclei is a foundational step in many biological analyses, as most of the body's 30 trillion cells contain a nucleus that holds the DNA—a complete genetic program for each cell. By accurately locating nuclei, researchers can differentiate individual cells within a sample and study their responses to various treatments, thereby gaining insights into underlying biological processes. The automation of nuclei detection is key to expediting drug testing, potentially reducing the usual 10-year period required for new drug development.

### Evaluation

The competition is evaluated using the mean average precision (mAP) across a range of intersection-over-union (IoU) thresholds from 0.5 to 0.95. For each predicted nucleus, an IoU is calculated with the corresponding ground truth as follows:

$$
\\mathrm{IoU}(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}
$$

For a threshold \\( t \\) (ranging over 0.5, 0.55, \\( \\dots \\), 0.95), the precision is computed using:

$$
\\mathrm{Precision}(t) = \\frac{\\mathrm{TP}(t)}{\\mathrm{TP}(t) + \\mathrm{FP}(t) + \\mathrm{FN}(t)}
$$

Here, a true positive (TP) is counted when a predicted nucleus matches a ground truth nucleus with an IoU above the threshold, a false positive (FP) when no matching ground truth exists, and a false negative (FN) when a ground truth nucleus is missed. The average precision for an image is the mean of the precision values over all IoU thresholds:

$$
\\text{Average Precision} = \\frac{1}{|\\text{thresholds}|} \\sum_{t} \\frac{\\mathrm{TP}(t)}{\\mathrm{TP}(t) + \\mathrm{FP}(t) + \\mathrm{FN}(t)}
$$

The final competition score is the mean of these average precisions across all test images.

Additional notes:

- **Evaluation Metric**: mean average precision across intersection-over-union thresholds from 0.5 to 0.95
- **Interface File**: `deepevolve_interface.py`

### Dataset Description

The dataset comprises a large number of segmented nuclei images captured under various conditions—different cell types, magnifications, and imaging modalities (brightfield vs. fluorescence). This diversity is intended to test and challenge the generalization capability of detection algorithms.

Each image is associated with an ImageId and organized as follows:
- **/stage1_train/**: Contains training images and their corresponding annotated masks.
- **/stage1_test/**: Contains test images along with their annotated masks.

Within each image folder:
- The **images** subfolder contains the raw image file.
- The **masks** subfolder (available only in the training set) contains segmented masks for each nucleus. Each individual mask corresponds to one nucleus, and masks do not overlap.

Note that the second stage of the dataset will include images from new experimental conditions as well as images that are excluded from scoring. Submissions must be provided in run-length encoded format as detailed on the competition's evaluation page.

## Initial Idea

The initial approach applies a U-Net segmentation network for nuclei detection in microscopy images. The key steps involved are:

1. Resize raw images to 256×256 pixels.
2. Normalize images to have zero mean and unit variance.
3. Convert ground truth masks into unique integer labels using connected-component analysis.
4. Train the U-Net using the Adam optimizer for up to 100 epochs with a soft Dice loss. An early stopping mechanism is employed based on the validation Dice coefficient.
5. At inference time, generate probability maps from the model and apply a threshold of 0.5. Extract individual nuclei using connected component analysis on the thresholded output.

For further details and a practical implementation guide, refer to this [PyTorch Tutorial on DSB2018](https://www.kaggle.com/code/cloudfall/pytorch-tutorials-on-dsb2018).