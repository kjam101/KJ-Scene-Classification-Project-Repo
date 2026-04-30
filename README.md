<<<<<<< HEAD
=======
# KJ-Scene-Classification-Project-Repo
Repository for LIU Scene Classification Project
# Robust Scene Classification using Deep Learning

>>>>>>> 1344bbac2e6ffd29bd2423f9dd83a521d9f7d999
# Scene Classification with Multi-Task Learning

## 📌 Overview
This project explores scene classification using deep learning, with an emphasis on improving performance and understanding model behavior through multi-task learning.

A baseline convolutional neural network (CNN) is implemented using transfer learning (ResNet-18). The model is then extended into a multi-task framework that jointly predicts scene category and auxiliary environmental attributes such as day/night conditions, weather, and environment type.

---

## 🎯 Objectives
- Build a baseline scene classification model  
- Extend the model using multi-task learning  
- Evaluate the impact of auxiliary tasks on performance  
- Analyze model behavior using visualization techniques  

---

## 📊 Dataset
- **Dataset:** Intel Image Classification Dataset  
- **Classes:** Buildings, Forest, Glacier, Mountain, Sea, Street  

### Data Split
- Train: 11,230 images  
- Validation: 2,804 images  
- Test: 3,000 images  

---

## ⚙️ Setup

### Install Dependencies
```bash
pip install torch torchvision numpy matplotlib scikit-learn umap-learn
<<<<<<< HEAD
=======
```
>>>>>>> 1344bbac2e6ffd29bd2423f9dd83a521d9f7d999

1. Train Baseline Model: python src/base_model.py

2. Train Multi-Task Model: python src/train_multitask.py

3. Run UMAP Visualization: python src/umap_visualization.py

🧠 Multi-Task Learning Setup

The multi-task model shares a ResNet-18 backbone and includes multiple prediction heads:

Scene Classification (primary task)
Environment Type (built vs natural)
Day/Night (brightness + contrast heuristic)
Weather (rule-based grouping)

project-root/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── src/
│   ├── base_model.py
│   ├── train_multitask.py
│   ├── umap_visualization.py
│   ├── create_val_split.py
│   └── test_multitask_debug.py (optional)
│
├── models/
│
├── results/
│   ├── plots/
│   └── predictions/
│
├── class_names.txt
├── README.md
└── .gitignore

🛠️ Technologies Used
Python
PyTorch & Torchvision
NumPy
Matplotlib
scikit-learn
UMAP
GitHub
Kaggle


📊 Notes
Baseline model uses transfer learning with ResNet-18
Multi-task learning is implemented with shared feature extraction and multiple heads
Both manual and uncertainty-based loss weighting strategies are supported
UMAP is used to visualize learned feature representations
