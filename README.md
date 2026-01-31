# RCNN: IoT-Enabled Wearable Sensor Framework for Physical Activity Recognition

This repository provides the complete replication package for the research paper:

**“An IoT-Enabled Convolutional Recurrent Neural Network Framework for Wearable-Sensor-Based Physical Activity Recognition”**

The repository contains all code, configurations, notebooks, and experimental protocols required to reproduce the results reported in the paper.

---

## 📌 Overview

This work proposes an IoT-based framework that integrates wearable sensors with a Convolutional Recurrent Neural Network (CRNN) to accurately recognize physical activities. The framework combines spatial feature extraction using CNNs with temporal modeling via RNNs, enabling robust activity classification from multimodal wearable-sensor data.

The repository supports:
- Full reproducibility
- Benchmark-based evaluation (no human subjects involved)
- Fair comparison with multiple baseline models
- Statistical significance testing

---

## 📊 Dataset

The experiments are conducted using a **publicly available benchmark dataset**:

- **Wearable Sensor System for Physical Education**  
  Kaggle link:  
  https://www.kaggle.com/datasets/ziya07/wearable-sensor-system-for-physical-education  

📌 **Note:**  
No primary data were collected by the authors. All datasets are anonymized and publicly accessible.

---

## 🧠 Models Implemented

### Proposed Model
- **CRNN (Convolutional Recurrent Neural Network)**

### Baseline Models
- CNN  
- LSTM  
- GRU  
- CNN–LSTM  
- Transformer  
- Traditional ML models:
  - Support Vector Machine (SVM)
  - Random Forest (RF)

All models are trained and evaluated under identical experimental settings for fair comparison.

---
RCNN/
│
├── data/ # Raw and processed datasets
├── configs/ # JSON configuration files for all models
├── notebooks/ # Jupyter notebooks (step-by-step experiments)
├── models/ # Model definitions
├── training/ # Training and evaluation scripts
├── utils/ # Data loading, preprocessing, metrics
├── results/ # Tables, figures, logs
├── splits/ # Train/validation/test splits (1–5 folds)
├── reproducibility/ # Seeds, hardware specs, experiment protocol
│
├── README.md
├── requirements.txt
├── environment.yml
└── LICENSE

---

## 📓 Jupyter Notebooks (Execution Order)

| Notebook | Description |
|--------|-------------|
| 01_data_exploration.ipynb | Dataset analysis & statistics |
| 02_preprocessing.ipynb | Data cleaning & preprocessing |
| 03_train_crnn.ipynb | Training proposed CRNN |
| 04_train_cnn.ipynb | CNN baseline |
| 05_train_lstm.ipynb | LSTM baseline |
| 06_train_gru.ipynb | GRU baseline |
| 07_train_transformer.ipynb | Transformer baseline |
| 08_train_traditional_ml.ipynb | SVM & Random Forest |
| 09_evaluation_metrics.ipynb | Accuracy, F1, MCC, AUC |
| 10_confusion_matrix.ipynb | Confusion matrix plotting |
| 11_statistical_tests.ipynb | Wilcoxon signed-rank tests |
| 12_result_visualization.ipynb | Figures for publication |

---

## ⚙️ Configuration Files

All experiments are **configuration-driven** using JSON files located in `configs/`.

Example:
```json
{
  "model": "CRNN",
  "learning_rate": 0.1,
  "batch_size": 32,
  "epochs": 100,
  "dropout": 0.5,
  "optimizer": "Adam",
  "random_seed": 42
}

---

## 📓 Jupyter Notebooks (Execution Order)

| Notebook | Description |
|--------|-------------|
| 01_data_exploration.ipynb | Dataset analysis & statistics |
| 02_preprocessing.ipynb | Data cleaning & preprocessing |
| 03_train_crnn.ipynb | Training proposed CRNN |
| 04_train_cnn.ipynb | CNN baseline |
| 05_train_lstm.ipynb | LSTM baseline |
| 06_train_gru.ipynb | GRU baseline |
| 07_train_transformer.ipynb | Transformer baseline |
| 08_train_traditional_ml.ipynb | SVM & Random Forest |
| 09_evaluation_metrics.ipynb | Accuracy, F1, MCC, AUC |
| 10_confusion_matrix.ipynb | Confusion matrix plotting |
| 11_statistical_tests.ipynb | Wilcoxon signed-rank tests |
| 12_result_visualization.ipynb | Figures for publication |

---

## ⚙️ Configuration Files

All experiments are **configuration-driven** using JSON files located in `configs/`.

Example:
```json
{
  "model": "CRNN",
  "learning_rate": 0.1,
  "batch_size": 32,
  "epochs": 100,
  "dropout": 0.5,
  "optimizer": "Adam",
  "random_seed": 42
}
reproducibility/experiment_protocol.md
pip install -r requirements.txt
jupyter notebook
conda env create -f environment.yml
conda activate rcnn_env
jupyter notebook

## 📂 Repository Structure

