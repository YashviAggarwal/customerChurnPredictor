#  Telecom Customer Churn Prediction System

A Machine Learning web application that predicts whether a telecom customer is likely to churn based on customer demographics, account details, and subscribed services.

The project includes complete data preprocessing, exploratory data analysis, feature engineering, model comparison, and deployment using Flask.

---

# Features

- Customer churn prediction using Machine Learning
- Clean and interactive Flask web interface
- Churn probability score
- Risk level classification (High / Medium / Low)
- Displays top factors influencing the prediction
- Responsive UI built with Bootstrap
- End-to-end ML pipeline from preprocessing to deployment

---

##  Project Structure

```
Customer-Churn-Predictor/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── notebook/
│   └── Customer_Churn_Prediction.ipynb
│
└── dataset/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

##  Dataset

**IBM Telco Customer Churn Dataset**

The dataset contains customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Contract Type
- Internet Service
- Payment Method
- Monthly Charges
- Total Charges
- Customer Tenure

Target Variable:

**Churn**
- Yes
- No

---

## 🛠 Tech Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

### Backend

- Flask

### Frontend

- HTML
- CSS
- Bootstrap 5

---

##  Machine Learning Workflow

### 1. Data Cleaning

- Removed duplicate records
- Converted TotalCharges to numeric values
- Handled missing values
- Removed unnecessary columns

### 2. Exploratory Data Analysis

Visualizations include:

- Churn Distribution
- Contract Type vs Churn
- Internet Service vs Churn
- Monthly Charges Distribution
- Customer Tenure Distribution
- Correlation Heatmap
- Boxplots

### 3. Feature Engineering

Created additional features:

- HighMonthlyBill
- LongTermCustomer

### 4. Data Preprocessing

- One-Hot Encoding
- Train-Test Split
- Feature Scaling using StandardScaler

### 5. Model Training

The following models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest

### 6. Model Evaluation

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Model Performance

| Model | Accuracy |
|---------|---------|
| Logistic Regression | **80%** |
| Random Forest | 79% |
| Decision Tree | 72% |

Logistic Regression achieved the best overall performance and was selected as the final model.

---

##  Web Application

The Flask application allows users to:

- Enter customer information
- Predict customer churn
- View churn probability
- View customer risk level
- Understand the top factors contributing to the prediction

---

## ▶ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/customerChurnPredictor.git
```

Move into the project folder

```bash
cd customerChurnPredictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Flask application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

##  Future Improvements

- Hyperparameter tuning using GridSearchCV
- SHAP explainability
- ROC Curve and AUC visualization
- Docker containerization
- Cloud deployment
- User authentication and prediction history


---

##  If you found this project useful, consider giving it a star.