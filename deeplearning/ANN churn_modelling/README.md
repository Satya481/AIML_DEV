# Deep Learning for Bank Customer Churn Prediction

This project demonstrates how to build, train, and evaluate a deep learning model (ANN) for predicting bank customer churn. The solution is based on TensorFlow/Keras and includes all steps from preprocessing to model evaluation. This README provides a comprehensive overview of the methodology, implementation, and the deep learning concepts leveraged.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What is Churn Prediction?](#what-is-churn-prediction)
- [Why Deep Learning for Churn?](#why-deep-learning-for-churn)
- [Dataset Description](#dataset-description)
- [Deep Learning Concepts Used](#deep-learning-concepts-used)
- [Pipeline Steps](#pipeline-steps)
  - [1. Data Loading & Exploration](#1-data-loading--exploration)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Building the Neural Network](#4-building-the-neural-network)
  - [5. Training & Early Stopping](#5-training--early-stopping)
  - [6. Evaluation & Visualization](#6-evaluation--visualization)
- [Results](#results)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [References](#references)

---

## Project Overview

Bank customer churn prediction is a binary classification problem: will a customer exit (leave the bank) or stay? Churn modelling is critical for banks to identify at-risk customers and take proactive retention measures. This project uses a deep learning approach to accurately predict churn based on customer data.

---

## What is Churn Prediction?

**Churn prediction** is the process of identifying customers who are likely to leave a service provider. In banking, retaining customers is vital due to high acquisition costs and competitive pressures.

---

## Why Deep Learning for Churn?

Traditional machine learning models (like logistic regression, decision trees) are powerful, but may struggle with complex, nonlinear relationships in data. Deep learning, especially **Artificial Neural Networks (ANNs)**, excels at:

- Capturing intricate patterns and nonlinear dependencies.
- Learning high-level abstractions from raw features.
- Adapting to large datasets with many features.
- Handling complex feature interactions without explicit human engineering.

**In this project, ANN is chosen because:**
- The dataset contains both numerical and categorical features with potentially complex interactions.
- Churn is influenced by various subtle factors that neural networks can learn.
- Deep learning can generalize well with sufficient data and regularization.

---

## Dataset Description

- **File:** `Churn_Modelling.csv`
- **Features:** Customer demographics, account details, activity status.
- **Target:** `Exited` (1: churned, 0: retained)

| Feature           | Description                              |
|-------------------|------------------------------------------|
| CreditScore       | Credit score of customer                 |
| Geography         | Country                                  |
| Gender            | Male/Female                              |
| Age               | Age in years                             |
| Tenure            | Years with bank                          |
| Balance           | Account balance                          |
| NumOfProducts     | Number of bank products                  |
| HasCrCard         | Has credit card (1/0)                    |
| IsActiveMember    | Active status (1/0)                      |
| EstimatedSalary   | Estimated annual salary                  |

---

## Deep Learning Concepts Used

### 1. **Artificial Neural Networks (ANN)**
- Composed of interconnected layers of neurons.
- Each neuron applies a weighted sum and a nonlinear activation.

### 2. **Activation Functions**
- **ReLU (Rectified Linear Unit):** Used in hidden layers to introduce nonlinearity.
- **Sigmoid:** Used in output layer for binary classification.

### 3. **Dropout**
- Regularization technique to prevent overfitting by randomly "dropping" neurons during training.

### 4. **Early Stopping**
- Stops training when validation performance no longer improves, avoiding overfitting.

### 5. **Standardization**
- Scaling features to zero mean and unit variance for stable training.

### 6. **One-Hot Encoding**
- Categorical features (Geography, Gender) converted to numerical form.

### 7. **Model Evaluation**
- **Confusion Matrix:** Breakdown of true/false positives/negatives.
- **Accuracy Score:** Overall prediction accuracy.

---

## Pipeline Steps

### 1. Data Loading & Exploration
- Load CSV with pandas.
- Preview data and inspect features.

### 2. Feature Engineering
- Select relevant columns for input (`X`) and output (`y`).
- One-hot encode categorical features.

### 3. Data Preprocessing
- Concatenate encoded columns.
- Drop original categorical columns.
- Train/test split (20% train, 80% test).
- Standardize features.

### 4. Building the Neural Network
- **Input Layer:** 11 features.
- **First Hidden Layer:** 7 units, ReLU, Dropout(0.2).
- **Second Hidden Layer:** 6 units, ReLU, Dropout(0.2).
- **Output Layer:** 1 unit, Sigmoid.

### 5. Training & Early Stopping
- Compile with Adam optimizer, binary crossentropy loss.
- Train with batch size 10, up to 1000 epochs.
- Validation split (33%), early stopping on validation loss.

### 6. Evaluation & Visualization
- Model accuracy and loss plotted over epochs.
- Predictions made on test set.
- Confusion matrix and accuracy score calculated.

---

## Results

- **Accuracy:** ~83.7%
- **Confusion Matrix:**
  ```
  [[6160,  204],
   [1097,  539]]
  ```
- **Training History:** Both accuracy and loss curves indicate good learning and generalization.

---

## How to Run

### 1. Install Dependencies

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

### 2. Execute Notebook

Open `deeplearning/Churn_Modelling.ipynb` in Jupyter and run all cells sequentially.

---

## Requirements

- Python 3.x
- TensorFlow 2.x
- pandas
- numpy
- matplotlib
- scikit-learn

---

## References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Original Dataset Source (Kaggle)](https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers)

---

**Author:** Satya481  
**Repository:** [AIML_DEV](https://github.com/Satya481/AIML_DEV)