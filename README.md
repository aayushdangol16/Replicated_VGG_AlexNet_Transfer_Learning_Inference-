# Deep Learning Architectures: VGG, AlexNet, and Transfer Learning with ResNet50

This project implements the VGG and AlexNet architectures as defined in their corresponding research papers. The models are trained on the [Imagenette dataset](https://github.com/fastai/imagenette), a subset of the ImageNet dataset with 10 classes. Additionally, the project includes a transfer learning implementation using the ResNet50 model for binary classification and performs inference using Gardio for cat vs. dog classification.

## Features

- **VGG and AlexNet Implementation**:  
  Custom implementations of the VGG and AlexNet architectures, trained on the Imagenette dataset for multi-class classification (10 classes).
  
- **Transfer Learning with ResNet50**:  
  - The ResNet50 model is pre-trained on a large dataset (ImageNet) with 1,000 classes.  
  - All parameters of the ResNet50 model are frozen.  
  - The final fully connected layer is replaced and modified for binary classification.  
  - The last layer is trained specifically for cat vs. dog classification.

- **Inference with Gardio**:  
  The trained ResNet50 model is used to perform inference on images using the Gardio platform, classifying images as either cat or dog.

## Dataset Details

- **Imagenette**: A smaller subset of the ImageNet dataset containing 10 classes, used to train the VGG and AlexNet models.  
- **Cat-Dog Dataset**: A binary classification dataset used to fine-tune the last layer of the ResNet50 model.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/aayushdangol16/Replicated_VGG_AlexNet_Transfer_Learning_Inference-.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Perform inference on Gardio:
   ```bash
   cd Inference
   python index.py
