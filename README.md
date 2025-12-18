# 🧠 DeepFake Detection – Research-Oriented Implementations

This repository contains **research-grade implementations of state-of-the-art DeepFake detection methods**, developed in direct support of my main research paper:

> **A Dual-Modality Spatio-Temporal and Frequency Framework for Robust DeepFake Detection**

The goal of this repository is to provide **clean, modular, and reproducible PyTorch implementations** focusing on **spatio-temporal analysis** and **frequency-domain modeling**, two key aspects in robust DeepFake detection.

---

## 📄 Implemented Papers

### 1️⃣ Enhancing DeepFake Detection Performance with Spatiotemporal Texture and DeepFake Learning Feature Fusion

**My contribution:** Implementation of the core model components using **PyTorch**.

#### 🔹 Implemented Modules

* **LBP Module (Local Binary Pattern)**
  Extracts fine-grained local texture features to capture micro-level inconsistencies introduced by DeepFake manipulation.

* **GLCM Module (Gray-Level Co-occurrence Matrix)**
  Models statistical pixel-level dependencies to enhance texture representation across multiple spatial scales.

* **3D CNN Siamese Network**
  A dual-branch Siamese architecture designed for:

  * **Face regions**
  * **Background regions**

  Key characteristics:

  * Each branch processes a stack of **4 consecutive frames**
  * Frames are treated as **3D volumes**
  * Learns **spatio-temporal representations** capturing motion and structural inconsistencies

This design enables simultaneous analysis of facial dynamics and background motion, addressing limitations of purely face-centric approaches.

---

### 2️⃣ D2Fusion: Dual-Domain Fusion with Feature Superposition for DeepFake Detection

**My contribution:** Implementation of critical data and model components.

#### 🔹 Data Pipeline

* Custom **PyTorch DataLoader** supporting both spatial and frequency branches
* Task-specific **data augmentation** strategies to improve generalization

#### 🔹 Multi-Scale Facial Swap Module (MSFSM)

This module is designed to **emphasize artifact-prone facial regions**:

* Employs **randomly sized sliding windows** at multiple scales
* Adaptively focuses on regions most likely to contain DeepFake artifacts
* Computes **DASIM (Difference-Aware Similarity Map)** from paired real and fake images
* Applies DASIM-guided augmentation to amplify subtle manipulation traces

This approach encourages the model to learn **semantically meaningful manipulation cues** rather than superficial patterns.

#### 🔹 Model Architecture Implementation

##### 🟢 Spatial Domain Branch

* Feature extraction using **ResNet-34 blocks**
* Captures high-level spatial and structural facial features

##### 🟣 Frequency Domain Branch

* **Fine-Grained Spectral Attention Module**
* Uses **Discrete Cosine Transform (DCT)** to project images into the frequency domain
* Applies attention over frequency bands sensitive to manipulation artifacts

##### 🔁 Bidirectional Cross-Attention Module

* Enables information exchange between spatial and frequency branches
* Aligns complementary representations for robust decision-making

---

### 3️⃣ DeepFake Face Detection via Multi-Level Discrete Wavelet Transform and Vision Transformer

**My contribution:** Implementation of the frequency-domain feature extraction module.

#### 🔹 Multi-Level Frequency Feature Extraction Module

* Based on **Haar Wavelet Transform**
* Performs **multi-level decomposition** of facial images
* Separates:

  * **Low-frequency components** (global structure)
  * **High-frequency components** (fine details and artifacts)

This frequency modeling strategy plays a central role in exposing non-natural patterns and is **directly reused in my main research framework**.

---

## 📊 Datasets

The following benchmark datasets were used for training and evaluation:

* **FaceForensics++ (FF++)**
* **Celeb-DF**

These datasets enable fair comparison with existing state-of-the-art DeepFake detection methods.

---

## 🛠️ Technologies & Tools

* PyTorch
* Torchvision
* OpenCV
* NumPy
* Scikit-image

---

## 🎯 Repository Purpose

This repository serves as:

* An experimental backbone for my main paper
* A reference implementation for DeepFake detection research
* A foundation for future **dual-domain** and **spatio-temporal** modeling approaches

All components are implemented in a **modular fashion**, allowing independent reuse of spatial, temporal, and frequency-domain modules.

---

> **Note:** This repository is research-oriented and intended for academic use, experimentation, and reproducibility rather than production deployment.
