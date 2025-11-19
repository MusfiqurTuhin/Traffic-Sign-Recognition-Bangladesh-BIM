# 🇧🇩 Traffic Sign Recognition for Bangladesh (BIM 2024)

Official repository for the conference paper:  
**"A Comparative Analysis of Various Deep Learning Models for Traffic Signs Recognition from the Perspective of Bangladesh"**

---

## 🔗 Quick Links

- 📄 **Paper (Springer / DOI):** [https://doi.org/10.1007/978-981-99-8937-9_37](https://doi.org/10.1007/978-981-99-8937-9_37)  
- 📘 **Conference:** 2nd International Conference on Big Data, IoT and Machine Learning (BIM 2024)  
- 📦 **Dataset (BTSR-13) – Kaggle:** [https://www.kaggle.com/datasets/musfiqurtuhin/bangladeshi-traffic-signs-btsr-13](https://www.kaggle.com/datasets/musfiqurtuhin/bangladeshi-traffic-signs-btsr-13)  
- 🤗 **Dataset Mirror – HuggingFace:** [https://huggingface.co/datasets/musfiqurtuhin/BTSR-13](https://huggingface.co/datasets/musfiqurtuhin/BTSR-13)  

---

## 🧠 About the Project

South Asian traffic signs often suffer from **faded paint**, **visual clutter**, and **non-standard variations**, making Western datasets like GTSRB insufficient. To address this, I developed **BTSR-13**, a specialized dataset of **8,386 images across 13 local classes**, and benchmarked multiple deep learning architectures.

Our best-performing model, **Vision Transformer (ViT)**, achieved **99.91% accuracy**, demonstrating strong robustness in real Bangladeshi road environments.

---

## 🏆 Benchmark Summary

| Rank | Model | Accuracy | Strategy |
|------|-------|----------|----------|
| 🥇 | **Vision Transformer (ViT)** | **99.91%** | Fine-Tuning |
| 🥈 | **DenseNet201** | **99.86%** | Fine-Tuning |
| 🥉 | **Xception** | **99.54%** | Fine-Tuning |
| 4 | **InceptionV3** | **98.90%** | Fine-Tuning |
| 5 | **MobileNetV2** | **97.12%** | Fine-Tuning |

---

## 📂 Dataset Structure (BTSR-13)

```
BTSR-13/
├── train/          # 70% (5,863 images)
├── val/            # 20% (1,671 images)
└── test/           # 10% (852 images)
```

---

## 📁 Repository Structure

```
Traffic-Sign-Recognition-Bangladesh-BIM/
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   ├── 00_vit_champion_model.ipynb
│   ├── 01_densenet201_baseline.ipynb
│   └── utils.ipynb
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── eval.py
│   └── predict.py
├── configs/
│   ├── vit.yaml
│   └── densenet201.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── BTSR-13/
├── experiments/
│   └── vit_run/
├── assets/
│   └── sample_images/
└── scripts/
    ├── download_dataset.sh
    └── launch_training.sh
```

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/MusfiqurTuhin/Traffic-Sign-Recognition-Bangladesh-BIM.git
cd Traffic-Sign-Recognition-Bangladesh-BIM
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Best-Performing Model (ViT)

```bash
python src/train.py --config configs/vit.yaml
```

Or open the notebook directly:

```bash
jupyter notebook notebooks/00_vit_champion_model.ipynb
```

---

## 👥 Authors & Affiliations

### Bangladesh Army University of Science and Technology (BAUST), Saidpur
- Md. Mahbubur Rahman Tusher
- Hasan Muhammad Kafi
- Susmita Roy Rinky
- Muhiminul Islam

### United International University (UIU), Dhaka
- Md. Musfiqur Rahman

---

## 📝 Citation (Springer Format)

If you use the dataset or code, please cite:

```bibtex
@InProceedings{10.1007/978-981-99-8937-9_37,
  author="Tusher, Md. Mahbubur Rahman and Kafi, Hasan Muhammad and Rinky, Susmita Roy and Islam, Muhiminul and Rahman, Md. Musfiqur",
  editor="Arefin, M. Shamim and Kaiser, M. Shamim and Bhuiyan, Towhid and Dey, Nilanjan and Mahmud, Mufti",
  title="A Comparative Analysis of Various Deep Learning Models for Traffic Signs Recognition from the Perspective of Bangladesh",
  booktitle="Proceedings of the 2nd International Conference on Big Data, IoT and Machine Learning (BIM)",
  year="2024",
  publisher="Springer Nature Singapore",
  address="Singapore",
  pages="547--557",
  isbn="978-981-99-8937-9",
  doi="10.1007/978-981-99-8937-9_37"
}
```

---

## 📬 Contact

**Md. Musfiqur Rahman**
- **GitHub:** [https://github.com/MusfiqurTuhin](https://github.com/MusfiqurTuhin)
- **Email:** [your.email@example.com](mailto:your.email@example.com)

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

We extend our gratitude to all contributors and the open-source community for making this research possible.
