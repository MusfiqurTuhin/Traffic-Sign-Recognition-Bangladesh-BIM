\# 🧠 Model Implementation & Experiment Logs

This directory contains the complete source code for the comparative analysis presented in our BIM 2023 paper. The experiments were conducted in three phases: \*\*Baseline Training\*\* (from scratch), \*\*Transfer Learning\*\* (frozen ImageNet weights), and \*\*Fine-Tuning\*\* (optimization).

\#\# 🏆 Champion Model  
| File | Model | Accuracy | Description |  
| :--- | :--- | :--- | :--- |  
| \`00\_ViT\_Champion\_Model.ipynb\` | \*\*Vision Transformer (ViT)\*\* | \*\*99.91%\*\* | The state-of-the-art transformer model that outperformed all CNN architectures in recognizing local Bangladeshi traffic signs. Uses patch-based image processing. |

\---

\#\# 🔬 Deep Analysis Models (Step-by-Step)  
For our top-performing CNN architectures (DenseNet201 and Xception), we separated the training phases into distinct notebooks to isolate the impact of pre-training versus fine-tuning.

\#\#\# DenseNet201 (Runner Up \- 99.86%)  
| File | Phase | Description |  
| :--- | :--- | :--- |  
| \`01\_DenseNet201\_Baseline.ipynb\` | Baseline | Training from scratch without pre-trained weights. Serves as the control group. |  
| \`02\_DenseNet201\_TransferLearning.ipynb\` | Transfer Learning | Loaded with \`weights='imagenet'\` with layers frozen. |  
| \`03\_DenseNet201\_FineTuning.ipynb\` | \*\*Fine-Tuning\*\* | Unfreezing top layers (N=30) to adapt specifically to Bangladeshi road features. |

\#\#\# Xception  
| File | Phase | Description |  
| :--- | :--- | :--- |  
| \`06\_Xception\_Baseline.ipynb\` | Baseline | Training from scratch. |  
| \`07\_Xception\_TransferLearning.ipynb\` | Transfer Learning | Feature extraction using frozen ImageNet weights. |  
| \`08\_Xception\_FineTuning.ipynb\` | Fine-Tuning | Optimization of the Xception architecture for final classification. |

\#\#\# InceptionV3  
| File | Phase | Description |  
| :--- | :--- | :--- |  
| \`04\_InceptionV3\_Baseline.ipynb\` | Baseline | Training from scratch. |  
| \`05\_InceptionV3\_Full\_Pipeline.ipynb\` | Full Pipeline | Contains both the transfer learning and fine-tuning steps in a single execution flow. |

\---

\#\# 🧪 Comparative Study Pipelines  
To ensure a comprehensive benchmark, we evaluated 5 additional state-of-the-art architectures. These notebooks contain the \*\*Full Pipeline\*\* (Data Preprocessing $\\to$ Transfer Learning $\\to$ Fine-Tuning $\\to$ Evaluation) in a single file.

| File | Architecture | Purpose |  
| :--- | :--- | :--- |  
| \`09\_NASNetLarge\_Pipeline.ipynb\` | \*\*NASNet Large\*\* | Testing Neural Architecture Search efficiency on local data. |  
| \`10\_EfficientNetB2\_Pipeline.ipynb\` | \*\*EfficientNetB2\*\* | Evaluating accuracy vs. parameter efficiency. |  
| \`11\_ResNet101\_Pipeline.ipynb\` | \*\*ResNet101\*\* | Testing deep residual learning performance. |  
| \`12\_VGG19\_Pipeline.ipynb\` | \*\*VGG19\*\* | Evaluating a classic deep architecture. |  
| \`13\_MobileNetV2\_Pipeline.ipynb\` | \*\*MobileNetV2\*\* | Testing feasibility for mobile/edge deployment. |

\#\# ⚙️ Technical Setup  
\* \*\*Framework:\*\* TensorFlow / Keras  
\* \*\*Hardware:\*\* Trained on NVIDIA T4 GPU (via Google Colab).  
\* \*\*Data Loading:\*\* Uses \`ImageDataGenerator\` with real-time augmentation (shear, zoom, horizontal flip).  
\* \*\*Loss Function:\*\* Categorical Crossentropy.  
\* \*\*Optimizer:\*\* Adam (Learning rate dynamically adjusted).

\#\# 📊 Key Findings  
1\.  \*\*Transformers vs CNNs:\*\* ViT (\`00\`) demonstrated that attention mechanisms generalize better on this specific dataset than traditional convolutions.  
2\.  \*\*Transfer Learning Impact:\*\* Models initialized with ImageNet weights (\`02\`, \`07\`) converged 40% faster than baselines (\`01\`, \`06\`).  
3\.  \*\*Fine-Tuning Necessity:\*\* In all cases, the Fine-Tuning step (unfreezing layers) resulted in a 1-3% accuracy boost, critical for safety-critical applications like traffic sign recognition.  
