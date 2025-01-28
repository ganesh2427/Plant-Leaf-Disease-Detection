# Plant-Leaf-Disease-Detection

## Overview
This project focuses on the detection and classification of plant leaf diseases using advanced convolutional neural network (CNN) architectures. The primary goal is to identify and categorize diseases affecting plant leaves, enabling early intervention to improve agricultural productivity and minimize losses. The project incorporates state-of-the-art deep learning techniques, including CBAM (Convolutional Block Attention Module) and compound scaling, to enhance model performance.



## Features
- Disease detection and classification using images of plant leaves.
- Implementation of various CNN architectures:
    - AlexNet
    - VGG16
    - EfficientNet
- Integration of CBAM for improved feature attention.
- Application of compound scaling to optimize model performance and efficiency.
- Developed using TensorFlow and supporting Python libraries.

## Technologies Used

- Programming Language: Python
- Framework: TensorFlow, Keras
- CNN Architectures: AlexNet, VGG16, EfficientNet
- Attention Mechanism: CBAM (Convolutional Block Attention Module)
- Optimization: Compound Scaling
- Libraries:
    - NumPy
    - Pandas
    - Matplotlib/Seaborn (for visualization)
    - Scikit-learn (for preprocessing and evaluation)




## Dataset

- The dataset consists of labeled images of plant leaves categorized by type and disease.
- Preprocessing steps:
    - Resizing images to a uniform size.
    - Data augmentation techniques such as rotation, flipping, and zooming to improve model generalization.
    - Splitting the dataset into training, validation, and testing sets.


## Project Workflow

1. Data Preparation:

    - Load and preprocess the dataset.

    - Perform data augmentation to enhance model generalization.

2. Model Selection:

    - Implement multiple CNN architectures (AlexNet, VGG16, EfficientNet).

    - Enhance models using CBAM for attention.

    - Optimize model performance using compound scaling.

3. Training and Evaluation:

    - Train models on the processed dataset using TensorFlow.

    - Evaluate models on validation and testing sets.

    - Compare performance metrics such as accuracy, precision, recall, and F1-score.

4. Visualization:

    - Plot training and validation loss/accuracy.

    - Generate heatmaps for attention visualization (CBAM).

## Results
    - Achieved high accuracy in detecting and classifying plant leaf diseases.
    - Comparative performance of different architectures:

        - AlexNet: Baseline performance.
        - VGG16: Improved feature extraction.
        - EfficientNet: Optimal trade-off between performance and efficiency.

    - Enhanced model interpretability using CBAM.


## Installation

1. Clone the repository:
2. Install the required dependencies:
3. Run the project:

## Folder Structure

    plant-leaf-disease-detection/
        ├── data/                 # Dataset folder
        ├── models/               # Pre-trained models and architectures
        ├── notebooks/            # Jupyter notebooks for experimentation
        ├── scripts/              # Python scripts for training and evaluation
        ├── results/              # Generated results and visualizations
        └── README.md            # Project documentation

## Future Scope

- Extend the dataset to include more plant species and diseases.
- Deploy the model as a web or mobile application for real-time disease detection.
- Explore other advanced architectures like Vision Transformers (ViT).

## Acknowledgments

- TensorFlow and Keras documentation.
- Research papers on CBAM and compound scaling.
- Datasets and community contributions to open-source projects.










