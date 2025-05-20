# 🩺 Pneumonia Detection from Chest X-Rays

This project applies transfer learning and an ensemble of deep learning models to detect pneumonia from chest X-ray images. It combines EfficientNet-B4, DenseNet-201, and VGG11 using soft-voting to improve diagnostic accuracy.

## 📂 Project Structure

```
├── chest_xray_data/         # Raw dataset (not tracked in Git)
├── notebooks/               # Jupyter notebooks for training and evaluation
├── models/                  # Saved PyTorch model weights
├── requirements.txt         # Dependencies
└── README.md                # You're here!
```

## 🧠 Models Used

- ✅ **EfficientNet-B4** (`tf_efficientnet_b4_ns`)
- ✅ **DenseNet-201**

Both models were fine-tuned using PyTorch/TIMM with custom classification heads for binary classification.

## 🔬 Dataset

- **Source**: [Chest X-ray Pneumonia Dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Structure**:
  ```
  chest_xray_data/
  ├── train/
  ├── val/
  └── test/
  ```

> ⚠️ This folder is excluded from version control via `.gitignore`.

## 🚀 Training

Use the notebook below to train EfficientNet-B4:

> [`notebooks/training_efficientnet.ipynb`](notebooks/training_efficientnet.ipynb)

It includes:
- Data augmentation and normalization
- Transfer learning setup
- Validation loss tracking and checkpoint saving
- Final evaluation on the test set

## 🧪 Inference (Ensemble)

To run ensemble predictions using soft voting:

> [`notebooks/ensemble_demo.ipynb`](notebooks/ensemble_demo.ipynb)

Steps:
- Loads EfficientNet and DenseNet models
- Averages their softmax outputs
- Predicts class label
- Reports final test accuracy

## 🧾 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
