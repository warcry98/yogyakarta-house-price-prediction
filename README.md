# Yogyakarta House Price Prediction

Deployed in streamlit cloud https://yogyakarta-house-price-prediction-k8kftr6funsmjwec5yd3mp.streamlit.app/

This project predicts **house prices in Yogyakarta, Indonesia** using classical machine learning models.
It is built as a **final project for the Machine Learning Zoomcamp** and demonstrates an end-to-end ML workflow:
data preparation, model training, evaluation, comparison, and deployment.

The project compares:
- **Linear Regression** (interpretable baseline)
- **XGBoost (Gradient Boosting)** (non-linear, higher-performance model)

Both models are deployed in a single **Streamlit application** with an interactive model selector.

---

## 📊 Dataset

- **Name:** Yogyakarta Housing Price Indonesia  
- **Source:** Kaggle  
- **Author:** pramudyadika  
https://www.kaggle.com/datasets/pramudyadika/yogyakarta-housing-price-ndonesia/code

The dataset contains real housing data with:
- Numerical features (e.g. size, number of rooms)
- Categorical features (e.g. location, house type)
- Target variable: **house price**

---

## 🧠 Models

### 1. Linear Regression (Baseline)
- Used as an interpretable reference model
- Preprocessing:
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features
- Purpose:
  - Establish baseline performance
  - Explain feature influence via coefficients

### 2. XGBoost Regressor (Gradient Boosting)
- Chosen for strong performance on tabular data
- Captures non-linear relationships and feature interactions
- Preprocessing:
  - Numerical features passed through without scaling
  - OneHotEncoder for categorical features

---

## ⚙️ Hyperparameter Tuning

XGBoost hyperparameters are tuned automatically using **RandomizedSearchCV**:

- Search performed **only on the training set**
- 5-fold cross-validation
- Optimization metric: Mean Absolute Error (MAE)

Tuned parameters include:
- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`

The best parameters are saved and displayed in the Streamlit app.

---

## 🔁 Cross-Validation

To estimate generalization performance:
- A single **train/validation split** (80/20) is used
- **5-fold cross-validation** is applied **only on the training set**
- Both models use the same folds for fair comparison

This avoids data leakage while providing robust performance estimates.

---

## 📈 Evaluation Metrics

Models are evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

Additional diagnostics:
- **Residual plots** to check error distribution
- **Metric improvement deltas** (XGBoost vs Linear Regression)

Metric deltas explicitly quantify how much XGBoost improves over the baseline.

---

## 🖥️ Streamlit Application

The deployed Streamlit app allows users to:
- Select a model (Linear Regression or XGBoost)
- Input house features via sidebar controls
- Generate real-time price predictions
- Compare models side-by-side
- View residual plots and tuned parameters
- Download predictions as a CSV file

---

## 🚀 How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python train.py

# Run Streamlit app
streamlit run app.py