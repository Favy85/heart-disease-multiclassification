#  Heart Disease Multiclassification  
Deep Learning | Machine Learning | Python | TabNet | FTTransformer | CNN | MLP  

This project predicts **five stages of coronary heart disease** using advanced deep learning and machine learning models.  
The goal is to build a robust multiclass classification system using a complete pipeline:
data cleaning → preprocessing → feature engineering → resampling → modelling → evaluation.



---

##  **Project Overview**

This project builds and compares four major models to classify heart disease stages:

- **MLP (Multi-Layer Perceptron)**
- **1D Convolutional Neural Network (CNN)**
- **TabNet Classifier**
- **FT-Transformer (attention-based deep model)**
  
The workflow includes:
- Handling missing values  
- Removing outliers using Isolation Forest  
- Feature selection with RFE + Random Forest  
- Variance thresholding  
- Balancing data using SMOTETomek  
- Hyperparameter tuning for all four models  
- Saving all preprocessors and models for future use  

This project demonstrates full **end-to-end deep learning modelling**, **traditional ML engineering**, and **deployment-ready preprocessing pipelines**.

---

##  **Dataset**

- **Source:** Kaggle — *Heart Disease UCI Dataset*  
- **Target variable:** `num` (5-class disease severity: 0 = healthy → 4 = critical)  
- **Features:** Demographic, clinical, ECG, exercise, and heart activity variables  

Due to licensing, the dataset is **not uploaded here**, but can be downloaded directly from Kaggle:  
🔗 [https://www.kaggle.com/datasets/ronitf/heart-disease-uci](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data)  

Place the dataset inside a folder named `data/` before running the notebook.

---

##  **Technologies & Libraries**

- **Python**
- **Scikit-learn**
- **TensorFlow / Keras**
- **PyTorch + Skorch**
- **TabNet**
- **Imbalanced-learn (SMOTETomek)**
- **Matplotlib / Seaborn**
- **Joblib for model saving**

---

##  **Data Preprocessing Steps**

### ✔ Missing Value Handling  
- Numerical features → *Iterative Imputer (MICE)*  
- Categorical features → *Most Frequent Imputer*

### ✔ Encoding  
- One-Hot Encoding (drop first)

### ✔ Outlier Removal  
- Isolation Forest (5% contamination)

### ✔ Feature Selection  
- **RFE (Random Forest)** → select 15 important features  
- **Variance Thresholding** → remove low-variance predictors

### ✔ Class Rebalancing  
- SMOTETomek to correct 5-class imbalance  
- Before & After comparison plots included

### ✔ Scaling  
- StandardScaler fitted on balanced data

All preprocessors saved as `.pkl` files.

---

##  **Models Implemented**

### 🔸 **1. Multi-Layer Perceptron (MLP)**
- Tuned using RandomizedSearchCV  
- Tested multiple architectures + activations  

### 🔸 **2. 1D Convolutional Neural Network (CNN)**
- Built using Keras  
- Tuned using **KerasTuner Hyperband**  
- Input shaped into (samples, features, 1) format

### 🔸 **3. TabNet Classifier**
- Tuned via RandomizedSearchCV  
- Designed for structured/tabular data  
- Excellent feature interpretability

### 🔸 **4. FTTransformer (PyTorch + Skorch)**
- Attention-based deep architecture  
- Hyperparameters tuned with scikit-learn wrapper  

---

## 📊 **Results**

### **Model Comparison Table**

| Model | Accuracy | F1-score | AUC-ROC | MCC |
|-------|----------|----------|---------|------|
| **MLP** | 0.88 | 0.88 | 0.97 | 0.85 |
| **CNN** | 0.80 | 0.79 | 0.94 | 0.75 |
| **TabNet** | 0.75 | 0.75 | 0.91 | 0.70 |
| **FTTransformer** | 0.84 | 0.84 | 0.96 | 0.81 |

### **Per-Class (F1 Score)**  
| Class | MLP | CNN | TabNet | FTTransformer |
|-------|------|-------|---------|-----------------|
| Healthy (0) | 0.86 | 0.82 | 0.79 | 0.84 |
| Mild (1) | 0.76 | 0.61 | 0.55 | 0.66 |
| Moderate (2) | 0.87 | 0.75 | 0.70 | 0.84 |
| Severe (3) | 0.91 | 0.79 | 0.73 | 0.87 |
| Critical (4) | 0.95 | 0.95 | 0.93 | 0.95 |

### **Key Insight**
👉 **MLP performed best overall**, while FTTransformer showed highly competitive results.  
👉 Critical cases (Class 4) were predicted with the highest accuracy across all models.

---

## 🖼 **Visualisations Included**

---

## 🔠 **1. Categorical Feature Distributions**

<p align="center">
  <img src="images/Bar plots of categorical features 1.png" width="48%" />
  <img src="images/Bar plots of categorical features 2.png" width="48%" />
</p>

**Summary:**  
I analysed all categorical variables using count plots to understand the distribution of key clinical features such as **sex**, **chest pain type**, **resting ECG results**, **exercise-induced angina**, **ST slope**, and **thalassemia status**.  
This highlighted imbalances, dominant categories, and clinically relevant patterns.  
These insights informed correct one-hot encoding and guided later balancing and feature engineering steps.

---

## 📈 **2. Numerical Feature Distributions**

<p align="center">
  <img src="images/Histograms of numerical features.png" width="90%" />
</p>

**Summary:**  
Numerical variables were examined using histograms and KDE overlays to assess distribution shape, skewness, and potential noise.  
This step revealed skewed variables such as **oldpeak** and **ca**, helping guide decisions on **scaling**, **imputation**, and **outlier treatment** before modelling.

---

## 📦 **3. Outlier Detection (Before Cleaning)**

<p align="center">
  <img src="images/Box Plot in detecting Outliers.png" width="85%" />
</p>

**Summary:**  
A combined boxplot visualised outliers in features like **chol**, **trestbps**, and **oldpeak**.  
These observations justified using **Isolation Forest** to remove extreme values, improving data quality and model stability.

---

## 🤖 **4. Feature Importance (RFE + Random Forest)**

<p align="center">
  <img src="images/RFE and Random Forest for Feature Selection.png" width="85%" />
</p>

**Summary:**  
Recursive Feature Elimination identified the most informative predictors including **ca**, **thalach**, **chol**, **age**, **oldpeak**, and specific encoded categories.  
This dimensionality reduction helped improve model interpretability and training efficiency.

---

## 📊 **5. Class Distribution Before & After Balancing**

<p align="center">
  <img src="images/Class Distribution Before Balancing.png" width="45%" />
  <img src="images/Class Distribution After Balancing.png" width="45%" />
</p>

**Class Labels:**  
- **0 – Healthy**  
- **1 – Mild Disease**  
- **2 – Moderate Disease**  
- **3 – Severe Disease**  
- **4 – Critical Disease**

**Summary:**  
Before balancing, the dataset was heavily skewed, with *Healthy (0)* dominating and *Critical (4)* severely underrepresented.  
After applying **SMOTETomek**, all classes became evenly distributed.  
This prevented majority-class bias during model training and ensured fair evaluation across all disease stages.

---

## 📈 **6. Model Performance Comparison**

<p align="center">
  <img src="images/Model Performance Comparison.png" width="85%" />
</p>

**Summary:**  
All four models (MLP, CNN, TabNet, FTTransformer) were evaluated using Accuracy, F1-score, AUC-ROC, and MCC.  
The **MLP model achieved the strongest overall performance**, followed closely by FTTransformer.  
These comparisons provided a clear understanding of which algorithms handled the multiclass structure most effectively.


