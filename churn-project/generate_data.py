"""
generate_data.py
Generates a realistic telecom customer churn dataset for demonstration.
"""

import numpy as np
import pandas as pd

def generate_churn_dataset(n=5000, seed=42):
    np.random.seed(seed)

    tenure = np.random.exponential(scale=30, size=n).clip(1, 72).astype(int)
    monthly_charges = np.random.normal(65, 20, n).clip(20, 120)
    total_charges = tenure * monthly_charges * np.random.uniform(0.9, 1.1, n)

    contract = np.random.choice(["Month-to-month", "One year", "Two year"],
                                 p=[0.55, 0.25, 0.20], size=n)
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"],
                                         p=[0.35, 0.45, 0.20], size=n)
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        p=[0.35, 0.22, 0.22, 0.21], size=n
    )
    tech_support = np.random.choice(["Yes", "No", "No internet"], p=[0.29, 0.49, 0.22], size=n)
    online_security = np.random.choice(["Yes", "No", "No internet"], p=[0.28, 0.50, 0.22], size=n)
    paperless_billing = np.random.choice(["Yes", "No"], p=[0.59, 0.41], size=n)
    senior_citizen = np.random.choice([0, 1], p=[0.84, 0.16], size=n)
    dependents = np.random.choice(["Yes", "No"], p=[0.30, 0.70], size=n)
    partner = np.random.choice(["Yes", "No"], p=[0.48, 0.52], size=n)
    phone_service = np.random.choice(["Yes", "No"], p=[0.90, 0.10], size=n)
    multiple_lines = np.where(phone_service == "No", "No phone service",
                              np.random.choice(["Yes", "No"], p=[0.42, 0.58], size=n))
    num_services = np.random.randint(1, 7, size=n)

    # Churn probability driven by realistic signals
    churn_prob = (
        0.05
        + 0.30 * (contract == "Month-to-month")
        + 0.15 * (internet_service == "Fiber optic")
        + 0.10 * (payment_method == "Electronic check")
        + 0.08 * (tech_support == "No")
        + 0.07 * (online_security == "No")
        - 0.25 * (tenure > 24)
        - 0.10 * (tenure > 48)
        + 0.05 * (monthly_charges > 80)
        + 0.05 * senior_citizen
        - 0.05 * (num_services > 4)
        + np.random.normal(0, 0.05, n)
    ).clip(0.02, 0.90)

    churn = (np.random.uniform(size=n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID": [f"TLC-{10000+i}" for i in range(n)],
        "gender": np.random.choice(["Male", "Female"], size=n),
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "Contract": contract,
        "MonthlyCharges": monthly_charges.round(2),
        "TotalCharges": total_charges.round(2),
        "NumServices": num_services,
        "Churn": churn
    })

    return df


if __name__ == "__main__":
    df = generate_churn_dataset()
    df.to_csv("data/telco_churn.csv", index=False)
    print(f"Dataset saved: {len(df)} rows, churn rate = {df['Churn'].mean():.1%}")
