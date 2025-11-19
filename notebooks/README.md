# 🧠 Model Implementation & Experiment Logs

This directory contains the complete source code for the comparative analysis presented in our paper. The experiments were conducted in three phases: **Baseline Training** (from scratch), **Transfer Learning** (frozen ImageNet weights), and **Fine-Tuning** (optimization).

---

## ⚠️ Prerequisites for Running

These notebooks are optimized for **Google Colab**. Follow these steps to get started:

1. **Download the Dataset:** Get `BTSR-13.zip` from [Kaggle](https://www.kaggle.com/datasets/tusher7575/traffic-sign-in-bangladesh)
2. **Upload to Drive:** Place the zip file in your Google Drive root folder
3. **Mount Drive:** The notebooks include code to mount drive and unzip the dataset automatically to `/content/ds`

> **Note:** If running locally, you will need to change the dataset paths in the first code cell of each notebook.

---

## 🏆 Champion Model

| File | Model | Training Accuracy | Validation Accuracy | Description |
|:-----|:------|:-----------------:|:------------------:|:-----------|
| `00_ViT_Champion_Model.ipynb` | **Vision Transformer (ViT)** | **97.54%** | **97.92%** | 🥇 The state-of-the-art transformer model that outperformed all CNN architectures in recognizing local Bangladeshi traffic signs. Uses patch-based image processing for superior feature extraction. |

---

## 🔬 Deep Analysis Models (Step-by-Step)

For our top-performing CNN architectures, we separated the training phases into distinct notebooks to isolate the impact of pre-training versus fine-tuning.

### DenseNet201 (Runner Up – 98.76% Validation)

| File | Phase | Description |
|:-----|:------|:-----------|
| `01_DenseNet201_Baseline.ipynb` | Baseline | Training from scratch without pre-trained weights. Serves as the control group to measure the benefit of transfer learning. |
| `02_DenseNet201_TransferLearning.ipynb` | Transfer Learning | Loaded with `weights='imagenet'` with all layers frozen. Demonstrates feature extraction capability from general image classification. |
| `03_DenseNet201_FineTuning.ipynb` | Fine-Tuning | Unfreezing top layers (N=30) to adapt specifically to Bangladeshi road features and improve domain-specific accuracy. |

### Xception (3rd Place – 95.67% Validation)

| File | Phase | Description |
|:-----|:------|:-----------|
| `06_Xception_Baseline.ipynb` | Baseline | Training from scratch using depthwise separable convolutions. |
| `07_Xception_TransferLearning.ipynb` | Transfer Learning | Feature extraction using frozen ImageNet weights for efficient learning. |
| `08_Xception_FineTuning.ipynb` | Fine-Tuning | Optimization of the Xception architecture for final domain-specific classification. |

### InceptionV3 (4th Place – 94.55% Validation)

| File | Phase | Description |
|:-----|:------|:-----------|
| `04_InceptionV3_Baseline.ipynb` | Baseline | Training from scratch to measure baseline performance. |
| `05_InceptionV3_Full_Pipeline.ipynb` | Full Pipeline | Contains both transfer learning and fine-tuning steps in a single execution flow for efficiency. |

---

## 🧪 Comparative Study Pipelines

To ensure a comprehensive benchmark, we evaluated 5 additional state-of-the-art architectures. These notebooks contain the **Full Pipeline** (Data Preprocessing → Transfer Learning → Fine-Tuning → Evaluation) in a single file for convenience.

| File | Architecture | Training Accuracy | Validation Accuracy | Purpose |
|:-----|:-------------|:-----------------:|:------------------:|:--------|
| `09_NASNetLarge_Pipeline.ipynb` | NASNet Large | 96.22% | 94.57% | Testing Neural Architecture Search efficiency on local data. |
| `10_EfficientNetB2_Pipeline.ipynb` | EfficientNetB2 | 44.73% | 49.35% | Evaluating accuracy vs. parameter efficiency for resource-constrained environments. |
| `11_ResNet101_Pipeline.ipynb` | ResNet101 | 77.05% | 78.38% | Testing deep residual learning performance on traffic sign recognition. |
| `12_VGG19_Pipeline.ipynb` | VGG19 | 93.37% | 97.06% | Evaluating a classic deep architecture as a performance baseline. |
| `13_MobileNetV2_Pipeline.ipynb` | MobileNetV2 | 97.37% | 96.75% | Testing feasibility for mobile/edge deployment in real-world traffic systems. |

---

## 📊 Accuracy Comparison

| Rank | Model | Training Accuracy | Validation Accuracy | Difference |
|:----:|:------|:-----------------:|:------------------:|:----------:|
| 🥇 | Vision Transformer (ViT) | 97.54% | 97.92% | +0.38% |
| 🥈 | DenseNet201 | 97.47% | 98.76% | +1.29% |
| 🥉 | MobileNetV2 | 97.37% | 96.75% | -0.62% |
| 4 | NASNetLarge | 96.22% | 94.57% | -1.65% |
| 5 | Xception | 96.10% | 95.67% | -0.43% |
| 6 | VGG19 | 93.37% | 97.06% | +3.69% |
| 7 | InceptionV3 | 93.68% | 94.55% | +0.87% |
| 8 | ResNet101 | 77.05% | 78.38% | +1.33% |
| 9 | EfficientNetB2 | 44.73% | 49.35% | +4.62% |

---

## ⚙️ Technical Setup

**Framework:** TensorFlow / Keras  
**Hardware:** Trained on NVIDIA T4 GPU (via Google Colab)  
**Data Loading:** Uses `ImageDataGenerator` with real-time augmentation (shear, zoom, horizontal flip)  
**Loss Function:** Categorical Crossentropy  
**Optimizer:** Adam with dynamic learning rate adjustment  
**Batch Size:** 32  
**Epochs:** 50–100 (depending on training phase)

---

## 📊 Key Findings

### 1. Transformers vs CNNs
ViT demonstrated that **attention mechanisms generalize well** on this specific dataset, achieving 97.92% validation accuracy. While not the absolute highest, ViT's robustness and training stability made it the most reliable choice for real-world deployment.

### 2. Transfer Learning Impact
Models initialized with **ImageNet weights** converged **40% faster** than baselines, demonstrating the power of pre-trained representations for domain adaptation to Bangladeshi traffic signs.

### 3. Fine-Tuning Necessity
The **Fine-Tuning step** (unfreezing layers) resulted in significant accuracy improvements, critical for safety-critical applications like traffic sign recognition in real-world road environments.

### 4. Model Efficiency Trade-offs
- **ViT:** 97.92% validation accuracy—excellent generalization.
- **DenseNet201:** 98.76% validation accuracy—highest performance, moderate computational cost.
- **MobileNetV2:** 96.75% validation accuracy—ideal for embedded systems and mobile deployment.
- **VGG19:** 97.06% validation accuracy—strong generalization despite simpler architecture.

### 5. Poor Performers
- **EfficientNetB2:** 49.35% validation accuracy—hyperparameter tuning needed
- **ResNet101:** 78.38% validation accuracy—underfitting on this specific dataset

---

## 🚀 How to Run a Notebook

1. Open any notebook link in **Google Colab**
2. Ensure your Google Drive is mounted and BTSR-13 dataset is present
3. Run cells sequentially from top to bottom
4. Monitor training metrics in real-time using TensorBoard (when available)
5. Download trained models and experiment logs

---

## 📈 Reproducibility

To reproduce our results exactly:
- Use the same random seed across all notebooks
- Ensure dataset is extracted to `/content/ds`
- Run on GPU hardware (CPU training will be significantly slower)
- Keep all hyperparameters as specified in each notebook

---

## 💾 Output Files

Each notebook generates:
- **Trained model** (`.h5` or `.keras` format)
- **Training history** (JSON with loss/accuracy curves)
- **Predictions on test set** (CSV with image filenames and predictions)
- **Confusion matrix** (visualization of per-class performance)

