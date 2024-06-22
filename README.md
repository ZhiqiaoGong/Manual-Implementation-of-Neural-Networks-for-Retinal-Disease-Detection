# Retinal Disease Classification Using Deep Learning

## Overview

This project aims to classify retinal diseases from Optical Coherence Tomography (OCT) images using various deep learning models. The models used include Simple CNN, U-Net, U-Net with pretrained ResNet weights, EfficientNet, and DenseNet. The goal is to evaluate the performance of these models in diagnosing conditions such as Choroidal Neovascularization (CNV), Diabetic Macular Edema (DME), and Drusen, with the objective of improving diagnostic accuracy and efficiency.


## Abstract

Retinal diseases such as CNV, DME, and Drusen are major causes of vision impairment. Manual analysis of OCT images for these conditions is time-consuming and prone to errors. This project explores the use of deep learning models to automate and improve the accuracy of retinal disease diagnosis. By implementing and evaluating models like Simple CNN, U-Net, U-Net with pretrained ResNet weights, EfficientNet, and DenseNet, we demonstrate the potential of advanced neural network architectures in medical image classification.

## Models and Methods

1. **Simple CNN**
   - Basic convolutional neural network architecture.
   - Training accuracy: 74.14%
   - Validation accuracy: 70.00%
   - Test accuracy: 73.11%

2. **U-Net**
   - Advanced architecture for medical image segmentation.
   - Training accuracy: 90.89%
   - Validation accuracy: 96.42%
   - Test accuracy: 96.46%

3. **U-Net with Pretrained ResNet Weights**
   - Hybrid model combining U-Net and pretrained ResNet layers.
   - Training accuracy: 93.11%
   - Validation accuracy: 96.78%
   - Test accuracy: 96.76%

4. **EfficientNet**
   - Model balancing performance and computational efficiency.
   - Training accuracy: 90.47%
   - Validation accuracy: 96.14%
   - Test accuracy: 95.54%

5. **DenseNet**
   - Dense connectivity to enhance feature propagation and reuse.
   - Training accuracy: 87.97%
   - Validation accuracy: 98.02%
   - Test accuracy: 98.87%

## Dataset

The dataset used in this project is sourced from Kaggle and contains 83,605 OCT images categorized into four classes: NORMAL, CNV, DME, and DRUSEN. The images are organized into training, validation, and test sets.

## Results

The performance of each model was evaluated based on accuracy metrics for training, validation, and test datasets. DenseNet achieved the highest test accuracy at 98.87%, indicating its effectiveness in classifying retinal diseases from OCT images.

## Conclusion

This project demonstrates the potential of deep learning models in automating and improving the accuracy of retinal disease diagnosis. The comparison of different models highlights the trade-offs between accuracy and computational efficiency, providing insights into the selection of appropriate models for medical image analysis tasks.

## Installation

To run this project, ensure you have Python and the following libraries installed:

- Pytorch
- NumPy
- OpenCV
- Matplotlib

Install the required libraries using:

```bash
pip install torch torchvision numpy scikit-learn matplotlib pillow
# Retinal-Disease-Classification-Using-Deep-Learning
```


## Usage
1. **Prepare the Dataset**:
   Download the OCT dataset from Kaggle and organize it into the following structure:
```css
OCT2017/
  ├── train/
  │   ├── CNV/
  │   ├── DME/
  │   ├── DRUSEN/
  │   └── NORMAL/
  ├── val/
  │   ├── CNV/
  │   ├── DME/
  │   ├── DRUSEN/
  │   └── NORMAL/
  └── test/
      ├── CNV/
      ├── DME/
      ├── DRUSEN/
      └── NORMAL/
```
2. **Run the Jupyter Notebook**:
   Open the ece285finalcode.ipynb notebook and execute the cells to preprocess the data, train the models, and evaluate their performance.

3. **Train and Evaluate Models**:
   Follow the instructions in the notebook to train the models (Simple CNN, U-Net, U-Net with pretrained ResNet weights, EfficientNet, DenseNet) and evaluate their performance on the test set.
