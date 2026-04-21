
# 1. Introduction
Pneumonia remains a critical global health challenge, particularly in pediatric populations and resource-constrained environments. Chest X-ray (CXR) imaging is the primary diagnostic modality due to its affordability and accessibility. However, interpreting CXRs requires significant radiological expertise, and inter-observer variability can lead to misdiagnoses.

While deep learning models, specifically Convolutional Neural Networks (CNNs), have demonstrated expert-level performance in automated pneumonia detection, their clinical deployment is hindered by a critical vulnerability: **"Shortcut Learning."** Recent studies have shown that standard CNNs often solve the classification task by latching onto spurious correlations—such as hospital-specific text markers, patient positioning artifacts, or scanner differences—rather than learning the underlying pathology. This results in models that perform exceptionally well on the training data but fail catastrophically when tested on images from different hospitals or scanners.

In this work, we propose a novel framework, **Anatomically Constrained Consistency Learning (ACCL)**, to address this reliability gap. By integrating domain knowledge (lung anatomy) directly into the learning process via a dual-branch architecture, we compel the model to verify its predictions against an anatomically masked version of the input. This ensures that the decision-making process is causally linked to the lung regions, significantly enhancing the model's robustness and trustworthiness for clinical application.

# 2. Background and Related Work
### 2.1 Deep Learning in Medical Imaging
Early applications of deep learning in radiology primarily utilized transfer learning with architectures like VGG16, ResNet, and DenseNet. Rajpurkar et al. (CheXNet) demonstrated that a 121-layer DenseNet could exceed the performance of practicing radiologists on the ChestX-ray14 dataset. Similarly, Kundu et al. proposed an ensemble of GoogLeNet, ResNet-18, and DenseNet-121, achieving high accuracy on the Kermany dataset. However, these methods typically treat the input image as a generic 2D signal, ignoring the specific anatomical structure of the chest.

### 2.2 Shortcut Learning and Robustness
The phenomenon of "Shortcut Learning," identified by Geirhos et al., describes the tendency of deep neural networks to learn the simplest possible solution to minimize loss, often relying on low-level texture or background cues rather than high-level semantic valid features. In the context of COVID-19 and pneumonia detection, Maguolo et al. showed that CNNs could achieve high accuracy even when the lung fields were completely removed, proving that the models were relying entirely on background artifacts (e.g., text labels, cables).

### 2.3 Consistency Regularization
To combat this, consistency regularization techniques have been employed, primarily in semi-supervised learning. The core idea is that a model's prediction should remain stable under various perturbations (e.g., rotation, noise). Our approach extends this by introducing **anatomical consistency**: the model's feature representation of a full X-ray should be mathematically consistent with its representation of the isolated lungs.

# 3. System Architecture and Design
The proposed system is designed as a dual-branch neural network that enforces feature consistency. The architecture comprises three main components:

1.  **Automated Anatomical Masking Module:** A computer-vision-based preprocessing pipeline that automatically segments the lung fields from the raw X-ray used during training.
2.  **Shared Feature Extractor (Backbone):** A ResNet-18 network that processes both the original image and the masked image. The weights are shared between the two branches to ensure they map to the same feature space.
3.  **Consistency Regularization Head:** A loss function component that penalizes the divergence between the feature vectors of the original and masked images.

This design ensures that the model cannot "cheat" by looking at the background, as the background is absent in the masked branch. If the model relies on background tags in the original image, the features will differ significantly from the masked image features (where tags are removed), incurring a high consistency loss.

# 4. IV. Methodology and Algorithmic Design
### 4.1 Automated Lung Segmentation
We implement a robust, unsupervised segmentation algorithm using classic Computer Vision techniques (OpenCV) to generate lung masks without requiring a separate, heavy segmentation network:
*   **Preprocessing:** The input image is converted to grayscale and Gaussian blurred ($5 \times 5$ kernel) to reduce noise.
*   **Otsu's Binarization:** An adaptive threshold is applied to separate the foreground (body) from the background.
*   **Morphological Opening:** Erosion followed by Dilation removes small artifacts (e.g., cables, text).
*   **Contour Analysis:** We identify the largest dark regions in the chest cavity, corresponding to the lungs, and generate a binary mask $M$.

### 4.2 Dual-Branch Training Strategy
Let $x$ be the input image and $y$ be the label. We generate an anatomically masked image $x_{mask} = x \odot M$.
The shared backbone $f(\cdot)$ processes both inputs:
$$ h_{orig} = f(x) $$
$$ h_{mask} = f(x_{mask}) $$

### 4.3 Objective Function
The total loss $L_{total}$ is a weighted sum of the Classification Loss ($L_{CE}$) and the Consistency Loss ($L_{Cons}$):
$$ L_{total} = L_{CE}(C(h_{orig}), y) + L_{CE}(C(h_{mask}), y) + \lambda \cdot || h_{orig} - h_{mask} ||^2 $$

Where:
*   $C(\cdot)$ is the classifier head (fully connected layers).
*   $|| \cdot ||^2$ is the Mean Squared Error (MSE) ensuring feature alignment.
*   $\lambda$ is a hyperparameter balancing the two objectives.

# 5. Threat Model and Security Consideration
In the context of medical AI, the primary "threat" is not a malicious adversary, but the **distributional shift** and **spurious correlations** inherent in medical data.

**Threat: Shortcut Learning (The "Lazy" Model)**
The threat model assumes the neural network is an optimizer that seeks the path of least resistance.
*   **Attack Vector:** Hospital-specific watermarks (e.g., "PORTABLE," "L"), patient support devices, or variations in X-ray beam intensity.
*   **Vulnerability:** Standard CNNs will overfit to these artifacts. For example, if all "Pneumonia" patients are scanned with a portable machine (showing a "PORTABLE" tag), the model learns to detect the tag, not the disease.

**Defense: Anatomically Constrained Consistency**
Our ACCL framework acts as a defense mechanism against this threat:
*   **Security Guarantee:** By enforcing $h_{orig} \approx h_{mask}$, we mathematically guarantee that the features used for classification are present in the lung region.
*   **Robustness:** The model becomes invariant to background noise. Even if a malicious actor (or a new hospital protocol) adds text to the background, the consistency check forces the model to ignore it.

# 6. Experimental Evaluation
### 6.1 Experimental Setup
*   **Dataset:** The Kermany Chest X-Ray dataset (5,856 images).
*   **Validation:** 5-Fold Cross-Validation to ensure statistical significance.
*   **Implementation:** PyTorch framework, trained on NVIDIA GPU.
*   **Baselines:** We compare against a standard ResNet-18 (Vanilla) and a VGG16 model.

### 6.2 Metrics
We evaluate the model using:
*   **Accuracy & AUC:** Standard performance metrics.
*   **Consistency Score:** The L2 distance between feature vectors of original vs. masked images (lower is better).
*   **OOD (Out-of-Distribution) Robustness:** Performance on a "corrupted" test set where we artificially add noise/blocks to the background to simulate scanner artifacts.
