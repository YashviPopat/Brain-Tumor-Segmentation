
## Table of contents
1) [Overview](#Overview)

2) [Features](#Features)

3) [Results](#Results)

4) [Requirements](#Requirements)

5) [User Interface](#UserInterface)



## Brain Tumor Segmentation

This project focuses on brain tumor segmentation using three advanced deep learning models: U-Net, V-Net, and SegNet. The primary objective was to compare the accuracy and performance of these models in identifying and segmenting brain tumors from medical images. Through extensive analysis, U-Net was found to provide the most precise and reliable tumor segmentation results. The purpose of this project is to facilitate early detection of brain tumors, enabling timely diagnosis and treatment, thereby helping to mitigate the adverse effects of tumors and improve patient outcomes.




 

## Features

- Early Detection of Tumors
- Segmented Brain Tumors
- Medical Image Processing
- Deep Learning Integration
- Real-Time Performance



## Dataset


1) https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
2) https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation
## Code Files

1. U-Net: https://github.com/YashviPopat/Brain-Tumor-Segmentation/blob/fb33196c11a63931cb90997b5a2b28e9e309c23c/UNet_Brain_Tumor_Tutorial.ipynb

2. SegNet: https://github.com/YashviPopat/Brain-Tumor-Segmentation/blob/fb33196c11a63931cb90997b5a2b28e9e309c23c/Segnet_Brain_Tumor_Segmentation.ipynb

3. V-Net: https://github.com/YashviPopat/Brain-Tumor-Segmentation/blob/fb33196c11a63931cb90997b5a2b28e9e309c23c/V_Net.ipynb
## Results

![Segmented_Tumor]((https://github.com/user-attachments/assets/45ec460f-6003-4bbc-8a6c-0b02b7b7204b))





## Requirements
1. Programming Language and Environment
Python 3.8+
Jupyter Notebook (optional, for running the code interactively)

2. Libraries and Frameworks
TensorFlow 2.8+ or PyTorch (based on the implementation)
Keras (if using TensorFlow backend)
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
OpenCV
SciPy

3. Dataset
Brain MRI dataset (e.g., BRATS dataset or other annotated brain tumor MRI datasets)
Ensure the dataset includes labeled images for segmentation.

4. Hardware Requirements
GPU-enabled system (recommended for training deep learning models):
NVIDIA CUDA-compatible GPU
CUDA Toolkit and cuDNN installed
Minimum 8 GB RAM (16 GB or higher recommended)

5. Tools and Dependencies
MediPy or SimpleITK (optional for medical image preprocessing)
Image processing libraries for augmentations (e.g., Albumentations or PIL)
Virtual environment tools (e.g., Conda or venv) for dependency isolation

6. Pre-trained Weights (Optional)
Pre-trained weights for U-Net, V-Net, or SegNet (if fine-tuning is intended).

7. Operating System
Compatible with Linux, Windows, or macOS.

## Gradio_Interface_UI

Code_File = https://github.com/YashviPopat/Brain-Tumor-Segmentation/blob/d1a3b09ed720e9cc3c11cb4f729d02e0e68fe55f/Gradio.ipynb