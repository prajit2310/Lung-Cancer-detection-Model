# Lung-Cancer-Detection-Model

## Overview
This project aims to automatically detect and classify lung cancer from chest medical images such as CT scans . The prediction framework is built around a **Convolutional Neural Network (CNN)** architecture designed to extract deep spatial features from grayscale images.  
The architecture includes multiple convolutional and pooling layers, followed by fully connected layers and dropout regularization, achieving strong performance even on variable-quality data.

## Features

**Exploratory Data Analysis (EDA):**  
- Visualization of image samples
-  class distribution
-  grayscale intensity histograms to understand data patterns and potential imbalances.

**Data Preprocessing & Augmentation:**  
- Grayscale image conversion and resizing to **256×256** resolution.  
- Standardization by normalizing pixel values.  
- Data augmentation using **random flips, rotations, zoom, and contrast adjustments** to improve model robustness and reduce overfitting.

**Model Architecture:**  
- A **Convolutional Neural Network (CNN)** with multiple convolutional and pooling layers for deep feature extraction.  
- Fully connected dense layers with **dropout regularization** to prevent overfitting.  
- Output layer with **softmax activation** for multi-class classification (four classes).

**Training Pipeline:**  
- Custom TensorFlow data pipeline using `tf.data.Dataset` for efficient loading and augmentation.  
- Adaptive learning rate with the **Adam optimizer** for faster convergence.

**Evaluation Metrics:**  
- Accuracy
- Loss curves visualization
- Potential metrics such as Precision, Recall, and F1-Score .
---

## Dataset

**Source:** CT scan image dataset stored in "Image_Dataset/train".

**Size:** 612 images across 4 classes.

**Image Format:** Grayscale medical images (JPEG, JPG, PNG).

**Target Classes:** 4 categories
- adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
- large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
- normal
- squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa

**Class Distribution :**
- adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib: 194 
- large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa: 115 
- normal: 148 
- squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa: 155
---

## Data Preprocessing

**Image Loading and Labeling:**  
Parsed image file paths from separate class folders and assigned numeric labels .

**Resizing and Normalization:**  
Resized all images to **256×256** resolution and normalized pixel values to the [0, 1] range for consistent input to the CNN.

**Grayscale Conversion:**  
Converted all images to **single-channel grayscale** to reduce complexity and focus on structural features.

**Data Augmentation:**  
Applied advanced augmentation techniques, including:  
- Random horizontal flips.  
- Random rotations (up to 5%).  
- Random zoom and contrast adjustments.  
These augmentations help improve model generalization and reduce overfitting.

**Batching and Prefetching:**   
Used TensorFlow's "tf.data" pipeline with **batching**, **shuffling**, and **prefetching** to ensure efficient and optimized data loading during training.
  
---

##  Model Architecture

**Convolutional Neural Network (CNN)**

- **Input Layer:**  
  Accepts grayscale images resized to **256×256×1**.

- **Convolutional & Pooling Layers:**  
  - **Layer 1:** 32 filters, 3×3 kernel, ReLU activation, followed by MaxPooling (2×2).  
  - **Layer 2:** 64 filters, 3×3 kernel, ReLU activation, followed by MaxPooling (2×2).  
  - **Layer 3:** 128 filters, 3×3 kernel, ReLU activation, followed by MaxPooling (2×2).

- **Flatten Layer:**  
  Converts 2D feature maps into a 1D vector for dense layers.

- **Fully Connected Layers:**  
  - Dense layer with 128 neurons, ReLU activation.  
  - Dropout layer (30%) to prevent overfitting.

- **Output Layer:**  
  Dense layer with **4 neurons**, softmax activation for multi-class classification.

- **Optimizer:**  
  Adam optimizer (learning rate = **0.0004**).

- **Loss Function:**  
  Sparse categorical cross-entropy.
---

## Evaluation Metrics

**Accuracy:** 83%  
**Macro Precision:** 86%  
**Macro Recall:** 85%  
**Macro F1-Score:** 85%  


