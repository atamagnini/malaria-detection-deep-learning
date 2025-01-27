# Malaria Cell Morphology Analysis and Subtype Classification using Deep Learning
## 
## Description

This project focuses on improving malaria diagnosis through advanced deep learning techniques. The goal is to detect and classify malaria cell types, both infected and uninfected, using a combination of Mask R-CNN for instance segmentation and MobileNet for multi-class classification. The pipeline also includes Faster R-CNN for object detection, VGG16 for classification, and U-Net for data augmentation to improve model robustness.

## Technologies Used

- Python
- PyTorch
- OpenCV
- Mask R-CNN
- Faster R-CNN
- MobileNet
- VGG16
- U-Net
- Scikit-learn
- Matplotlib

## File Descriptions

- code-segmentation.py: Implements instance segmentation using Mask R-CNN.
- data_prep.py: Preprocessing script that handles normalization, augmentation, and class balancing for the dataset.
- mobilenet.py: Classification model built using MobileNet for malaria cell types.
- object-detection-fastrcnn.py: Object detection code using Faster R-CNN for detecting cells.
- unet_imagetoimage.py: Implements image-to-image translation with U-Net for data augmentation.
- unet_imagetoimage_gen.py: Helper script for generating augmented images using U-Net.
- vgg16.py: Implements the VGG16 model for multi-class classification of cell types.
- project_report.pdf: Detailed report on the methodology, results, and analysis.

## Results

- MobileNet achieved significant accuracy in classifying malaria cell types, outperforming previous methods.
- Mask R-CNN showed improved detection and segmentation of cells, allowing for more precise diagnostics.
- Faster R-CNN and VGG16 were utilized to further enhance model performance for object detection and classification, respectively.

## Future Work

- Mask R-CNN will be trained further using augmented data for robust instance segmentation.
- Exploration of ensemble models and advanced data augmentation techniques.
- Deployment in clinical settings for real-time malaria diagnostics.
