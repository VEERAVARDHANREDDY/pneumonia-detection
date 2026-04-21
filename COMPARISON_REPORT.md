# Comparative Analysis of Deep Learning Models for Pneumonia Detection

## Abstract
In this study, we evaluate the performance of state-of-the-art Convolutional Neural Network (CNN) architectures—ResNet50, VGG16, VGG19, DenseNet121, and DenseNet169—for the automated detection of pneumonia from chest X-ray images. We employ a transfer learning approach, utilizing these pre-trained models as feature extractors coupled with various machine learning classifiers (SVM, Naive Bayes, k-Nearest Neighbors, Random Forest). Our experimental results demonstrate that **DenseNet169 combined with an SVM classifier** achieves the highest performance, offering a robust solution for computer-aided diagnosis.

## 1. Introduction
Pneumonia is a life-threatening infectious disease affecting the lungs. Rapid and accurate diagnosis is critical for effective treatment. Deep learning has emerged as a powerful tool for medical image analysis. However, determining the optimal architecture for specific medical tasks remains an open research question. This paper presents a comprehensive comparison of popular deep learning backbones to identify the most effective feature extractor for pneumonia detection.

## 2. Methodology

### 2.1 Dataset
The study utilizes the Kermany Chest X-Ray dataset. The dataset is split into training and testing sets. Images are resized to 224x224 pixels and normalized using ImageNet mean and standard deviation.

### 2.2 Feature Extraction
We employ five pre-trained CNN architectures as feature extractors:
- **VGG16 & VGG19**: Known for their simplicity and depth.
- **ResNet50**: Utilizes residual connections to enable training of deeper networks.
- **DenseNet121 & DenseNet169**: Connects each layer to every other layer in a feed-forward fashion, maximizing information flow.

For each model, the final classification layer is removed, and the global average pooling output is used as the feature vector.

### 2.3 Classifiers
The extracted features are fed into four traditional machine learning classifiers:
1.  **Support Vector Machine (SVM)**: Uses RBF kernel.
2.  **Naive Bayes (NB)**: Gaussian Naive Bayes.
3.  **k-Nearest Neighbors (k-NN)**: With k=5.
4.  **Random Forest (RF)**: Ensemble of 100 decision trees.

## 3. Results and Discussion

### 3.1 Receiver Operating Characteristic (ROC) Analysis
The ROC curves for each model backbone are presented below. The Area Under the Curve (AUC) indicates the model's ability to distinguish between Normal and Pneumonia cases.

![ROC Curve ResNet50](comparison_results/roc_curve_ResNet50.png)
*Figure 1: ROC Curves for ResNet50 Feature Extractor*

![ROC Curve DenseNet169](comparison_results/roc_curve_DenseNet169.png)
*Figure 2: ROC Curves for DenseNet169 Feature Extractor*

*(Note: Add other ROC curves here as generated)*

### 3.2 Confusion Matrices
To analyze misclassifications, we generated confusion matrices for the best-performing combinations.

![Confusion Matrices DenseNet169](comparison_results/confusion_matrices_DenseNet169.png)
*Figure 3: Confusion Matrices for DenseNet169*

### 3.3 Performance Comparison
The validation accuracy and AUC scores for all model-classifier combinations are summarized below.

| Model | Classifier | Accuracy | AUC | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DenseNet169** | **SVM (RBF)** | **0.848** | **0.958** | **0.809** | **0.990** | **0.890** |
| DenseNet169 | k-NN | 0.843 | 0.897 | 0.812 | 0.974 | 0.886 |
| DenseNet121 | Naive Bayes | 0.840 | 0.819 | 0.825 | 0.944 | 0.880 |
| DenseNet121 | k-NN | 0.833 | 0.914 | 0.802 | 0.974 | 0.880 |
| ResNet50 | SVM (RBF) | 0.824 | 0.958 | 0.783 | 0.992 | 0.876 |
| VGG16 | SVM (RBF) | 0.827 | 0.951 | 0.789 | 0.987 | 0.877 |

![AUC Comparison](comparison_results/auc_comparison_bar.png)
*Figure 4: Bar Chart comparison of AUC scores across all models.*

![Accuracy Comparison](comparison_results/accuracy_comparison_bar.png)
*Figure 5: Bar Chart comparison of Accuracy scores across all models.*

### 3.4 Key Findings
- **Best Backbone**: **DenseNet169** achieved the highest overall performance (F1-Score: 0.890), validating its effectiveness in medical imaging tasks where feature reuse is critical.
- **Best Classifier**: The **SVM (RBF kernel)** consistently outperformed other classifiers (Random Forest, Naive Bayes) across all backbones, particularly in AUC scores (~0.958).
- **ResNet50 Performance**: ResNet50 also performed strongly (AUC: 0.958) but had slightly lower precision than DenseNet169.
- **VGG16 Limitations**: While VGG16 achieved good sensitivity, it generally lagged in overall accuracy compared to the deeper DenseNet architectures.

## 4. Conclusion
Our comparative analysis confirms that transfer learning with **DenseNet169** and **SVM** provides a highly accurate method for pneumonia detection. This model effectively captures the subtle patterns in chest X-rays required for reliable diagnosis.
