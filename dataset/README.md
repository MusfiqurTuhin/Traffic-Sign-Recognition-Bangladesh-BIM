# 🇧🇩 BTSR-13 Dataset

## Dataset Overview

Due to the large size of the dataset (8,386 images, approx 300MB+), the raw image files are not hosted directly on GitHub. Instead, we utilize professional data hosting platforms (Kaggle & Hugging Face) to ensure high download speeds, version control, and DOI citation.

---

## ⬇️ Download Links

Please download the dataset from one of the official mirrors below:

| Platform | Link | Features |
|----------|------|----------|
| **Kaggle (Primary)** | [Download from Kaggle](https://www.kaggle.com/datasets/tusher7575/traffic-sign-in-bangladesh) | 🥇 Includes DOI, Visual Explorer, Kernel support |
| **Hugging Face** | [Download from HF](https://huggingface.co/datasets/musfiqurtuhin/BTSR-13) | 🚀 Fast Python loading via datasets library |

---

## 📂 Directory Structure

After downloading and extracting the `.zip` file, organize the folder as follows for the code to run without errors:

```
dataset/
├── train/                  # 5,863 images (70%)
│   ├── College in front/
│   ├── Crossroad/
│   ├── Left turn/
│   └── ... (13 classes)
│
├── val/                    # 1,671 images (20%)
│   ├── College in front/
│   └── ...
│
└── test/                   # 852 images (10%)
    ├── College in front/
    └── ...
```

---

## 🚦 Class Labels

The dataset contains the following **13 distinct classes** specific to Bangladeshi road environments:

1. College in front
2. Crossroad
3. Left turn
4. Market in front
5. Mosque in front
6. Pedestrian crossing
7. Rail crossing
8. Right turn
9. School in front
10. Side road left
11. Side road right
12. Speed breaker
13. Speed limit

---

## 🐍 Quick Load (Python)

If you use Hugging Face, you can load the data directly in your notebook without downloading the zip manually:

```python
from datasets import load_dataset

# Load from Hugging Face Hub
dataset = load_dataset("musfiqurtuhin/BTSR-13")

# Access splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']
```

### Alternative: Manual Loading

```python
import os
from PIL import Image
import numpy as np

def load_images_from_folder(folder_path):
    images = []
    labels = []
    
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(np.array(img))
                labels.append(class_name)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)

# Load train data
X_train, y_train = load_images_from_folder('dataset/train')
X_val, y_val = load_images_from_folder('dataset/val')
X_test, y_test = load_images_from_folder('dataset/test')

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
```

---

## 📊 Dataset Statistics

| Split | Images | Percentage |
|-------|--------|-----------|
| Training | 5,863 | 70% |
| Validation | 1,671 | 20% |
| Testing | 852 | 10% |
| **Total** | **8,386** | **100%** |

---

## 📝 Citation

If you use the BTSR-13 dataset, please cite the original paper:

```bibtex
@InProceedings{10.1007/978-981-99-8937-9_37,
  author="Tusher, M.M.R. and Kafi, H.M. and Rinky, S.R. and Islam, M. and Rahman, M.M.",
  title="A Comparative Analysis of Various Deep Learning Models for Traffic Signs Recognition from the Perspective of Bangladesh",
  booktitle="Proceedings of the 2nd International Conference on Big Data, IoT and Machine Learning (BIM)",
  year="2024",
  publisher="Springer Nature Singapore",
  doi="10.1007/978-981-99-8937-9_37"
}
```

---

## ⚖️ License & Usage

This dataset is provided for academic and research purposes. Ensure you comply with the terms of use on Kaggle and Hugging Face platforms.

---

## 🤝 Support

For issues or questions regarding the dataset, please:
- Open an issue on the GitHub repository
- Contact the authors directly
- Check the Kaggle dataset discussion section

