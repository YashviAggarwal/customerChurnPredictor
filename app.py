from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# ---------------------------------------------------------
# Load trained artifacts (created in your notebook)
# ---------------------------------------------------------
model = joblib.load("models/model.pkl")          # LogisticRegression
scaler = joblib.load("models/scaler.pkl")        # StandardScaler
columns = joblib.load("models/columns.pkl")      # list of training column names, in order

# Mean MonthlyCharges from training data, used for the HighMonthlyBill feature.
# Replace with your actual df["MonthlyCharges"].mean() from the notebook.
TRAINING_MEAN_MONTHLY_CHARGES = 64.76

# Fields the form doesn't collect — filled with the most common ("mode") value
# from the training set so the encoded input still matches the trained columns.
DEFAULTS = {
    "gender": "Male",
    "SeniorCitizen": "0",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "PaperlessBilling": "Yes",
}

# Friendly labels used to turn a raw column name into a human-readable reason
FRIENDLY_NAMES = {
    "Contract_Month-to-month": "Month-to-month contract",
    "Contract_One year": "One-year contract",
    "Contract_Two year": "Two-year contract",
    "InternetService_Fiber optic": "Fiber optic internet",
    "InternetService_DSL": "DSL internet",
    "InternetService_No": "No internet service",
    "PaymentMethod_Electronic check": "Pays by electronic check",
    "PaymentMethod_Mailed check": "Pays by mailed check",
    "PaymentMethod_Bank transfer (automatic)": "Pays by automatic bank transfer",
    "PaymentMethod_Credit card (automatic)": "Pays by automatic credit card",
    "TechSupport_No": "No tech support",
    "TechSupport_Yes": "Has tech support",
    "OnlineSecurity_No": "No online security",
    "OnlineSecurity_Yes": "Has online security",
    "HighMonthlyBill": "Above-average monthly bill",
    "LongTermCustomer": "Long-tenure customer",
    "tenure": "Customer tenure",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
}


def friendly(col_name):
    return FRIENDLY_NAMES.get(col_name, col_name.replace("_", " "))


def risk_level(probability):
    if probability >= 0.66:
        return "High Risk", "risk-high"
    elif probability >= 0.33:
        return "Medium Risk", "risk-medium"
    else:
        return "Low Risk", "risk-low"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()

    # Fill any fields not present on the form with training-set defaults
    for key, val in DEFAULTS.items():
        form_data.setdefault(key, val)

    input_df = pd.DataFrame([form_data])

    # Numeric fields must be cast explicitly — form values arrive as strings
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col])

    # Engineered features — must match exactly what was done in the notebook
    input_df["HighMonthlyBill"] = (
        input_df["MonthlyCharges"] > TRAINING_MEAN_MONTHLY_CHARGES
    ).astype(int)
    input_df["LongTermCustomer"] = (input_df["tenure"] >= 24).astype(int)

    # One-hot encode, then align to the exact training column order
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    label, css_class = risk_level(probability)

    # --- Top reasons: contribution = coefficient * scaled feature value ---
    # Positive contribution = pushes this specific customer toward churn.
    contributions = model.coef_[0] * input_scaled[0]
    contrib_series = pd.Series(contributions, index=columns)

    # Only consider features that are actually "on" or meaningfully non-zero
    # for this customer, so we don't surface irrelevant one-hot zero columns.
    active_mask = input_encoded.iloc[0] != 0
    active_contribs = contrib_series[active_mask]

    top_reasons = (
        active_contribs.sort_values(ascending=False)
        .head(3)
        .index.map(friendly)
        .tolist()
    )
    # Fall back gracefully if fewer than 3 features were "active"
    top_reasons = top_reasons if top_reasons else ["No strong risk factors found"]

    result = "Likely to Churn" if prediction == 1 else "Not Likely to Churn"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability * 100, 1),
        risk_label=label,
        risk_class=css_class,
        top_reasons=top_reasons,
        form_data=form_data,  # repopulate the form after submit
    )


if __name__ == "__main__":
    app.run(debug=True)