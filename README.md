# ❤️ Heart Disease Risk Prediction Using Machine Learning

## 📌 Project Overview

Heart disease risk prediction is a **machine learning classification project** designed to build a data-driven model for identifying whether a patient is likely to have heart disease based on available health-related attributes.

The project demonstrates an end-to-end machine learning workflow including **data exploration, preprocessing, feature transformation, class-imbalance handling, model preparation, and automated machine learning using AutoGluon**.

The primary objective is to transform raw healthcare data into a structured machine learning pipeline capable of supporting reliable binary classification.

---

## 🎯 Project Objective

The main objectives of this project are to:

* Analyze patient-related data and understand the distribution of the target variable.
* Clean and preprocess numerical and categorical features.
* Convert categorical variables into machine-readable numerical representations.
* Prevent data leakage by separating training and testing data before scaling.
* Handle class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**.
* Prepare the dataset for machine learning classification.
* Explore automated model training and model selection using **AutoGluon TabularPredictor**.

---

## 🏗️ Machine Learning System Architecture

The project follows a structured machine learning architecture that transforms raw healthcare data into a model-ready dataset through systematic **data ingestion, exploratory analysis, preprocessing, feature transformation, class balancing, and automated model development**.

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                    HEART DISEASE PREDICTION SYSTEM                         │
│                       ML PIPELINE ARCHITECTURE                             │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  01  DATA INGESTION                                                       │
│                                                                            │
│      Heart Disease Dataset (.CSV)  ───────────────►  Pandas DataFrame      │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  02  DATA UNDERSTANDING & EXPLORATORY ANALYSIS                            │
│                                                                            │
│      Dataset Shape  │  Features  │  Data Types  │  Descriptive Statistics │
│                                      │                                     │
│                                      ▼                                     │
│                         Target Distribution Analysis                       │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  03  DATA PREPROCESSING                                                   │
│                                                                            │
│      Duplicate Check ──► Feature Selection ──► Categorical Encoding        │
│                                                      │                     │
│                                                      ▼                     │
│                                                LabelEncoder                │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  04  FEATURE PREPARATION                                                  │
│                                                                            │
│                     Processed Dataset                                      │
│                           │                                                │
│                 ┌─────────┴─────────┐                                      │
│                 ▼                   ▼                                      │
│          Input Features (X)     Target Variable (y)                        │
│                              has_heart_disease                             │
│                 │                                                          │
│                 └─────────────► Train-Test Split                           │
│                                  80% / 20%                                 │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  05  FEATURE SCALING & CLASS BALANCING                                    │
│                                                                            │
│         Training Features ──► MinMaxScaler ──► SMOTE                       │
│                                                  │                         │
│                                                  ▼                         │
│                                      Balanced Training Dataset             │
│                                                                            │
│         Testing Features ───► MinMaxScaler                                 │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  06  AUTOMATED MACHINE LEARNING                                           │
│                                                                            │
│                     AutoGluon TabularPredictor                             │
│                                │                                           │
│                                ▼                                           │
│                   Automated Model Development                             │
│                                │                                           │
│                                ▼                                           │
│                    Model Training & Selection                              │
└────────────────────────────────────────────────────────────────────────────┘
```

### 🔄 Architecture Summary

**Data Source → Data Ingestion → EDA → Data Preprocessing → Feature Preparation → Train-Test Split → Feature Scaling → Class Balancing → AutoML → Model Training & Selection**

This layered architecture demonstrates a systematic approach to developing a **binary classification machine learning solution**, while maintaining clear separation between data preparation, feature transformation, imbalance handling, and model development.


## 🔍 Exploratory Data Analysis

Initial data analysis was performed to understand:

* Dataset dimensions
* Available features
* Numerical descriptive statistics
* Categorical descriptive statistics
* Dataset structure and data types
* Distribution of the target variable

The target variable used in this project is:

`has_heart_disease`

A target-distribution visualization was also created to examine the balance between the two prediction classes.

---

## 🧹 Data Preprocessing

### 1. Duplicate Handling

Duplicate records were checked during the preprocessing stage to improve overall data quality.

### 2. Feature Selection

The following features were removed from the modeling dataset:

* `patient_id`
* `wearable_owner`
* `family_history`
* `exercise_induced_angina`

### 3. Categorical Encoding

Categorical variables were transformed into numerical representations using:

`LabelEncoder`

This allows machine learning algorithms to process categorical information.

### 4. Feature and Target Separation

The dataset was separated into:

**Features (X)**
All predictor variables except the target.

**Target (y)**
`has_heart_disease`

---

## ✂️ Train-Test Split

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

A fixed `random_state` was used to support reproducibility.

---

## 📏 Feature Scaling

**MinMaxScaler** was applied to normalize feature values.

Importantly, the scaler was:

* Fitted only on the training data
* Applied to both training and testing data

This workflow helps prevent information from the test set from influencing preprocessing of the training data.

---

## ⚖️ Handling Class Imbalance with SMOTE

To improve representation of the minority class, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training dataset.

SMOTE generates synthetic samples for the minority class rather than simply duplicating existing observations.

This can help machine learning models learn more effectively when the target classes are imbalanced.

---

## 🤖 Automated Machine Learning

The project also explores **AutoGluon TabularPredictor** for automated machine learning.

AutoGluon can automate several parts of the modeling process, including training and comparing multiple candidate models.

```python
from autogluon.tabular import TabularDataset, TabularPredictor

label = "has_heart_disease"

predictor = TabularPredictor(label=label).fit(df)
```

---

## 🛠️ Technologies & Libraries

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| Python           | Programming language                   |
| Pandas           | Data manipulation                      |
| NumPy            | Numerical operations                   |
| Matplotlib       | Data visualization                     |
| Seaborn          | Statistical visualization              |
| Scikit-learn     | Preprocessing and train-test splitting |
| Imbalanced-learn | SMOTE implementation                   |
| AutoGluon        | Automated machine learning             |
| Google Colab     | Development environment                |

---

## 💡 Key Technical Concepts Demonstrated

This project demonstrates practical knowledge of:

* Machine Learning Classification
* Exploratory Data Analysis
* Data Cleaning
* Feature Selection
* Categorical Encoding
* Train-Test Splitting
* Feature Scaling
* Class Imbalance Handling
* SMOTE
* Automated Machine Learning
* Reproducible ML Workflows

---

## 📂 Suggested Repository Structure

```text
Heart-Disease-Prediction/
│
├── README.md
├── heartdisease_predictionmodel.py
├── requirements.txt
├── data/
│   └── heart_disease_risk_2026.csv
│
├── notebooks/
│   └── HeartDisease_PredictionModel.ipynb
│
└── images/
    └── target_distribution.png
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Heart-Disease-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Project

```bash
python heartdisease_predictionmodel.py
```

---

## 📈 Future Improvements

The project can be extended by:

* Comparing individual classification algorithms such as Logistic Regression, Decision Tree, Random Forest, and XGBoost.
* Performing hyperparameter optimization.
* Evaluating models using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
* Applying feature importance and explainability techniques such as SHAP.
* Building a reusable Scikit-learn preprocessing pipeline.
* Deploying the final model through a **Streamlit web application**.
* Adding model monitoring and versioning for an MLOps-oriented workflow.

---

## ⚠️ Disclaimer

This project is developed for **educational and machine learning portfolio purposes**. Predictions produced by the model should not be interpreted as professional medical diagnoses or medical advice.

---

## 👨‍💻 Author

**Mihir Jadhav**

Data Analytics | Machine Learning | Python | SQL | Power BI

GitHub: **MihirJ2002**

---

⭐ If you found this project useful, consider starring the repository.
