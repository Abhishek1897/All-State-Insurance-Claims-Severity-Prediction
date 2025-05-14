
# All State Insurance Claim Severity Prediction 🚗📉

This repository contains a complete machine learning pipeline to predict the **severity of insurance claims** using structured customer data. The project involves **exploratory data analysis, data preprocessing, feature selection, ensemble modeling, hyperparameter tuning, and result evaluation**.

---

## 📁 Project Structure

```
.
├── Ensemble_Machine_Learning_Project_All_State_Insurance_Claims_Severity_Prediction.ipynb
├── Ensemble_Machine_Learning_Project_All_State_Insurance_Claims_Severity_Prediction_Model_Loss_prediction.ipynb
├── train_features.sav
├── finalized_model.sav
├── prediction_output.csv
├── columns_to_drop.csv
├── *.sav (Label Encoders and Models)
└── README.md
```

---

## 📊 Objective

To build a robust machine learning model that can accurately **predict the severity (loss amount) of insurance claims** based on categorical and continuous features.

---

## 🧰 Tools & Libraries

- **Python**
- **pandas**, **numpy**
- **matplotlib**, **seaborn** for EDA
- **scikit-learn** for preprocessing, modeling, and evaluation
- **pickle** for saving models
- **RandomForestRegressor**, **GradientBoostingRegressor**, **VotingRegressor**

---

## 🔁 Workflow Overview

### 1. Data Acquisition
- Downloads training and test datasets from a public Amazon S3 link.

### 2. Data Preprocessing
- Handles missing values using SimpleImputer.
- Applies log transformation to the skewed 'loss' variable.
- Encodes categorical features with LabelEncoder.
- Treats outliers using IQR method.

### 3. Feature Engineering
- Drops low-variance, quasi-constant, and highly correlated features.
- Uses Chi-Squared test to remove redundant categorical variables.

### 4. Modeling
- Builds:
  - Base Random Forest Regressor
  - Tuned Random Forest with RandomizedSearchCV
  - Gradient Boosting Regressor

### 5. Evaluation
- Uses RMSE (Root Mean Squared Error) to compare model accuracy.
- Saves trained models for reuse.

### 6. Inference (2nd Notebook)
- Loads test data and saved model.
- Predicts loss on unseen data and saves comparison output (`prediction_output.csv`).

---

## 📈 Model Performance Metrics

- Metrics Used: **RMSE**, **Model Comparison**
- Models Saved: `finalized_model.sav`, `tunedmodel_rf.sav`, `basemodel_GBM.sav`

---

## 📂 Outputs

- `prediction_output.csv`: Contains actual vs predicted losses and the difference.
- `columns_to_drop.csv`: List of features removed during feature selection.
- `*.sav`: Trained models and encoders.

---

## 🔍 Authors & Credits

This project was developed as part of a machine learning case study on regression modeling and ensemble learning techniques.

---

## 📜 License

This project is released under the [MIT License](LICENSE).
