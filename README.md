# Customer Churn Prediction using Machine Learning

This project implements a **Customer Churn Prediction** system using machine learning techniques. The dataset used is the Telco Customer Churn dataset. The project includes data preprocessing, exploratory data analysis, feature encoding, handling class imbalance using SMOTE, and model training using Decision Tree, Random Forest, and XGBoost classifiers.

## 📂 Dataset

The dataset used is `WA_Fn-UseC_-Telco-Customer-Churn.csv`, which contains customer information and churn status. Important columns include:
- Customer demographic details (e.g., `gender`, `SeniorCitizen`, `Partner`)
- Subscription information (e.g., `Contract`, `PaymentMethod`)
- Usage metrics (e.g., `tenure`, `MonthlyCharges`, `TotalCharges`)
- Target variable: `Churn`

---

## 🚀 Features

✔ Data cleaning and handling missing values  
✔ Label encoding for categorical features  
✔ Exploratory data analysis with plots  
✔ Handling imbalanced classes using SMOTE  
✔ Model training using Decision Tree, Random Forest, and XGBoost  
✔ Model evaluation using accuracy, confusion matrix, and classification report  
✔ Saving and loading models with `pickle`

---
## 📊 Visualizations and Model Training Results

### Correlation Heatmap

(<img width="748" height="634" alt="Screenshot 2025-09-06 123603" src="https://github.com/user-attachments/assets/7a81c26c-0315-4d39-b64b-c2f464868fa4" />)

This heatmap shows the correlation between three numerical features:
- **tenure** and **TotalCharges** have a strong positive correlation (0.83), indicating that customers with longer tenure tend to accumulate more charges.
- **MonthlyCharges** is moderately correlated with **TotalCharges** (0.65).
- **tenure** and **MonthlyCharges** have a weak positive correlation (0.25).

Correlation values close to 1 indicate strong relationships, while values near 0 suggest weak or no correlation.

---

### Distribution of tenure
(<img width="559" height="375" alt="Screenshot 2025-09-06 123824" src="https://github.com/user-attachments/assets/35973f13-c97f-440b-bb70-9d5c3cee419b" />
)

This plot shows how customer tenure is distributed:
- There are significant numbers of customers with both short and long tenures.
- The mean and median lines highlight the central tendency of customer retention.
- The bimodal distribution suggests distinct customer segments—some new, some long-term.

---


### Distribution of MonthlyCharges

(<img width="565" height="377" alt="Screenshot 2025-09-06 123741" src="https://github.com/user-attachments/assets/86243c7a-d84f-4511-b082-ed3a6bfb7e4f" />
)

This plot displays the distribution of `MonthlyCharges` among customers:
- The data is spread across various charge ranges with noticeable peaks.
- The red dashed line represents the **mean**, while the green solid line represents the **median**.
- The skewness in the distribution suggests variability in customer plans and usage patterns.

---

### Distribution of Total Charges

(<img width="560" height="378" alt="Screenshot 2025-09-06 123747" src="https://github.com/user-attachments/assets/99e0c53b-e0ee-4106-b049-b7eb6f021de0" />
)

This plot displays the distribution of `TotalCharges` among customers:
- This chart shows a strong right-skewed distribution, indicating that most customers have lower total lifetime charges.
- The mean is pulled significantly higher than the median by a long tail of high-value, long-tenured customers.
- This pattern reflects a large number of new customers mixed with a smaller number of long-term ones.
---
### Box Plot of Monthly Charges
(<img width="705" height="461" alt="Screenshot 2025-09-06 123834" src="https://github.com/user-attachments/assets/efc69ab8-7032-4b6d-b509-9184b4cadec4" />
)
- The plot's central line shows the median monthly charge is approximately $70.
- The box represents the middle 50% of customers, who pay between ~$35 and ~$90 per month.
- It visualizes a wide spread of typical monthly charges across the customer base with no extreme outliers.

###  Model Training Results

(<img width="692" height="219" alt="Screenshot 2025-09-06 123910" src="https://github.com/user-attachments/assets/46282013-c930-4d9d-a6d8-058810d8486d" />
)

This output summarizes the cross-validation accuracy of three machine learning models used for churn prediction:

- **Decision Tree** achieved an accuracy of 78%.
- **Random Forest** performed the best with 84% accuracy.
- **XGBoost** also achieved a high accuracy of 83%.

These results highlight the effectiveness of ensemble methods like Random Forest in improving prediction performance.

---

### Model Performance Evaluation
(<img width="592" height="352" alt="Screenshot 2025-09-06 123917" src="https://github.com/user-attachments/assets/1b9fcac4-f7f6-4e91-9751-ba284ebacee9" />
)
- The model achieves an overall accuracy of approximately 78% in predicting customer churn.
- It reliably predicts customers who won't churn but struggles to identify those who will (58% recall for the churn class).
- The confusion matrix shows that 158 customers who actually churned were missed by the model.

###  Single Prediction Example
(<img width="394" height="108" alt="Screenshot 2025-09-06 123925" src="https://github.com/user-attachments/assets/b6ef8a44-1184-4e12-9543-a037b5cf5b9d" />
)
- The model's final prediction for this specific customer is "No Churn".
- This decision is made with high confidence, backed by an 83% probability score.
- The outcome is determined by selecting the class with the highest probability ("No Churn" > "Churn").



### 📂 Notes
- These visualizations help understand the dataset's structure and feature relationships.
- The models demonstrate the feasibility of predicting customer churn with high accuracy.
- Further improvements could include hyperparameter tuning, feature engineering, and model interpretability analysis.


## 📖 Key Code Snippets

### Load the dataset

```python
import pandas as pd
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocessing

df = df.drop(columns=["customerID"])
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)

# Label Encoding

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for column in df.select_dtypes(include='object').columns:
    df[column] = encoder.fit_transform(df[column])

# Hansling Class imbalance using SMOTE

from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Model training example (Random Forest)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Saving Model

import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

```

## 📦 Installation

1. Clone the repository:
git clone https://github.com/yourusername/customer-churn-prediction.git

2. Istall dependencies:
pip install -r requirements.txt


## 📊 Results

All models achieved high cross-validation accuracy scores (~100%), demonstrating that the dataset is suitable for classification tasks, though care must be taken regarding overfitting.
