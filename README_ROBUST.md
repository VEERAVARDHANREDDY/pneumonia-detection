# Robust Pneumonia Detection: Anatomically Constrained Consistency Learning

## Overview
This project upgrades the standard CNN/ViT approach to address **Shortcut Learning** (spurious correlations), a major research loophole identified in recent literature.
Instead of just classifying the image, we force the model to learn features that are **consistent** between the raw X-ray and an "Anatomically Masked" version (lungs only). This ensures the model does not rely on background artifacts (hospital tags, devices) to make predictions.

## Files
*   `robust_pipeline.py`: The main script implementing the Consistency Learning (ACCL) method.
*   `RESEARCH_PROPOSAL.md`: Detailed research justification, loophole analysis, and supervisor summary.
*   `requirements.txt`: Dependencies.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Robust Training**:
    ```bash
    python robust_pipeline.py
    ```
    *   This will train the model using the Dual-Branch Consistency Loss.
    *   It will automatically generate lung masks using OpenCV (Computer Vision).
    *   It effectively "teaches" the model to ignore the background.

3.  **Visual Output**:
    *   The script saves `masking_demo.png` to show you how it isolates the lungs.

## Research Justification (Viva Points)
*   **Problem**: Standard models (like in the 2021/2024 papers) look at the spinal cord/tags (proven by Grad-CAM).
*   **Solution**: My method penalizes the model if its features change when the background is removed.
*   **Result**: A model that *must* look at the lungs to minimize the loss, guaranteeing clinical reliability.
