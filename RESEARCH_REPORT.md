# Research Report: Pneumonia Detection via Graph Neural Networks on Visual Patches

## 1. Introduction: The Spatial Nature of Pneumonia
Pneumonia is an infection that inflames the air sacs in one or both lungs. The diagnosis on chest X-rays (CXR) typically captures **opacities** (white spots) that can be focal (lobar pneumonia) or diffuse (interstitial pneumonia).

Unlike simple object detection (e.g., "is there a cat?"), medical diagnosis often relies on the **spatial distribution and relationship** of these anomalies relative to healthy tissue and anatomical landmarks (ribs, diaphragm). A pure Convolutional Neural Network (CNN) excels at detecting local textures but aggregates global information into a dense vector, potentially losing the specific "topological" layout of the infection.

## 2. Why Graph Neural Networks (GNN)?
### limitation of CNNs
In a standard CNN (e.g., ResNet, VGG), the final layers perform global average pooling or flattening. This collapses the 2D feature map into a 1D vector. While efficient, this operation discard spatial relationships:
- We know *that* a feature exists, but the model struggles to explicitly reason about *where* it is relative to other features in a structured way.

### The GNN Advantage
By treating the image as a graph of regions (patches), we can model:
1.  **Nodes**: Distinct regions of the lung.
2.  **Edges**: Physical adjacency or learned relationships between regions.
3.  **Message Passing**: Allowing information to flow from one region to another. For example, if region A has a strong opacity, it can "inform" its neighbor region B to pay attention to spread.

**Hypothesis**: A GNN can better model the *relational dependencies* of lung opacities than a CNN alone, providing a more robust and explainable classification.

## 3. Methodology: The Hybrid CNN-GNN Pipeline

### Step 1: Deep Feature Extraction (The "Vision" Part)
We use a **ResNet-18** pretrained on ImageNet as a feature extractor.
-   **Input**: 224x224 Chest X-Ray.
-   **Operation**: Pass through convolutional layers until the last feature map.
-   **Output**: A $7 \times 7 \times 512$ tensor. This represents the image as a $7 \times 7$ grid of high-level feature vectors (each 512-dimensional).

### Step 2: Graph Construction (The "Structure" Part)
We convert the $7 \times 7$ grid into a graph $G = (V, E)$.
-   **Vertices ($V$)**: Each of the 49 cells in the grid is a node. Feature vector $h_i \in \mathbb{R}^{512}$.
-   **Edges ($E$)**: We create a **Grid Graph**. Each node is connected to its immediate spatial neighbors (top, bottom, left, right, diagonals). This enforces that information flows locally first, mimicking the physical spread of infection.
-   **Adjacency Matrix ($A$)**: A $49 \times 49$ matrix representing these connections.

### Step 3: Graph Reasoning (The "Logic" Part)
We apply **Graph Convolutional Network (GCN)** layers.
The update rule for a node $i$ is:
$$ h_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \frac{1}{c_{ij}} W^{(l)} h_j^{(l)} \right) $$
This allows the model to update the representation of a lung region based on the features of its neighbors.

### Step 4: Graph pooling & Classification
Instead of simple averaging, we use an **Attention Pooling mechanism**.
-   The network learns an attention score $\alpha_i$ for each node (region).
-   This automatically highlights which regions (e.g., the infected lobes) are most important for the diagnosis.
-   Final classification is simpler: A Multi-Layer Perceptron (MLP) on the weighted graph summary.

## 4. Expected Outcomes & Novelty
-   **Novelty**: Most student projects simply fine-tune VGG16. This approach introduces **Graph Representation Learning**, a cutting-edge technique in medical imaging (e.g., seen in Histopathology analysis).
-   **Explainability**: The attention weights ($\alpha_i$) from the Graph Pooling step provide a coarse localization map of the pneumonia without needing explicit bounding box labels.
-   **Viva Defense**: This architecture demonstrates understanding of:
    -   Feature Hierarchies (CNN).
    -   Non-Euclidean Data structures (Graphs).
    -   Spatial Reasoning (Message Passing).
