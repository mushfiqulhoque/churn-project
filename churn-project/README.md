<<<<<<< HEAD
# 📉 Customer Churn Prediction + Business Strategy

> **End-to-end ML pipeline** that predicts which telecom customers will churn, identifies top churn drivers, and quantifies the ROI of targeted retention campaigns.

---

## 🔍 Problem Statement

Customer churn costs telecom companies billions annually. Acquiring a new customer costs 5–7× more than retaining one. This project answers two questions:
- **Who is about to leave?** (Predictive ML)
- **What should we do about it?** (Business strategy & cost-benefit analysis)

---

## 📊 Results

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Logistic Regression | 0.758 | 0.764 |
| **Random Forest** | **0.759** | **0.764** |
| Gradient Boosting | 0.743 | 0.752 |

**Business Impact (test set):**
- 545 high-risk customers identified for intervention
- $286,800 revenue at risk from predicted churners
- $56,780 net ROI from targeted retention campaign (vs. blind outreach)

---

## 🧠 Key Churn Drivers

1. **Contract Type** — Month-to-month customers churn at 3× the rate of annual subscribers
2. **Tenure** — First 6 months are highest-risk; loyalty compounds over time
3. **Total Charges** — Accumulated billing correlates inversely with churn risk
4. **Payment Method** — Electronic check users show highest churn probability
5. **Internet Service** — Fiber optic customers churn more despite (or because of) premium pricing

---

## 📁 Project Structure

```
customer_churn_project/
├── data/
│   └── telco_churn.csv          # Generated synthetic dataset (5,000 customers)
├── src/
│   ├── generate_data.py         # Realistic synthetic data generator
│   └── churn_analysis.py        # Full ML pipeline
├── outputs/
│   ├── 01_eda_overview.png      # EDA charts
│   ├── 02_model_evaluation.png  # ROC, PR curves, confusion matrix
│   ├── 03_feature_importance.png
│   ├── 04_business_strategy.png # ROI analysis
│   └── high_risk_customers.csv  # Flagged customers for intervention
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python src/generate_data.py

# 3. Run full analysis pipeline
python src/churn_analysis.py
```

All charts and the high-risk customer list will appear in `/outputs/`.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | ML models, evaluation, preprocessing |
| `matplotlib`, `seaborn` | Visualization |
| `GradientBoostingClassifier` | Best-performing model |

---

## 📈 Methodology

1. **Data Generation** — Realistic synthetic telecom dataset with known churn signals
2. **EDA** — Churn by contract, tenure, charges, internet service, payment method
3. **Feature Engineering** — Encoded categoricals, derived features (`ChargePerMonth`, loyalty flags)
4. **Modeling** — Logistic Regression, Random Forest, Gradient Boosting with 5-fold cross-validation
5. **Evaluation** — ROC-AUC, Precision-Recall, Confusion Matrix
6. **Business Layer** — Risk tiering, optimal intervention threshold, ROI calculation

---

## 💡 Business Recommendations

Based on the model outputs:
- **Target Month-to-month customers in months 1–6** with upgrade offers to annual plans
- **Intervention threshold of 0.25** maximizes net ROI (545 customers, $56K net gain)
- **Electronic check users** are high-churn — nudge toward auto-pay with a small discount
- **Fiber optic churners** indicate service quality issues, not just price sensitivity

---

## 👤 Author

**Md Mushfiqul Hoque** — Data Analyst  
[LinkedIn](https://www.linkedin.com/in/md-mushfiqul-hoque-946220335/)
=======
# python_project
>>>>>>> dc28dd91749bd41d6cd0d4606036dbc3be4fed70
