"""
churn_analysis.py
End-to-end Customer Churn Prediction Pipeline
Author: [Md Mushfiqul Hoque]

Covers:
  - Data loading & cleaning
  - Exploratory Data Analysis (EDA) with business insights
  - Feature engineering
  - Model training: Logistic Regression, Random Forest, Gradient Boosting
  - Evaluation: ROC-AUC, Precision-Recall, Confusion Matrix
  - Feature importance & business interpretation
  - Cost-benefit analysis for churn intervention
  - High-risk customer targeting
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# ── Setup ──────────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

PALETTE = {"Churned": "#e74c3c", "Retained": "#2ecc71"}
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 130})

# ── 1. Load Data ───────────────────────────────────────────────────────────────
print("=" * 60)
print("  CUSTOMER CHURN PREDICTION PIPELINE")
print("=" * 60)

df = pd.read_csv("data/telco_churn.csv")
print(f"\n[1] Data loaded: {df.shape[0]:,} customers, {df.shape[1]} features")
print(f"    Churn rate: {df['Churn'].mean():.1%}")
print(f"    Null values: {df.isnull().sum().sum()}")

# ── 2. EDA ─────────────────────────────────────────────────────────────────────
print("\n[2] Running EDA...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Customer Churn — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.01)

# 2a. Churn distribution
ax = axes[0, 0]
counts = df["Churn"].value_counts()
labels = ["Retained", "Churned"]
colors = [PALETTE["Retained"], PALETTE["Churned"]]
wedges, texts, autotexts = ax.pie(
    [counts[0], counts[1]], labels=labels, colors=colors,
    autopct="%1.1f%%", startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for t in autotexts:
    t.set_fontsize(12); t.set_fontweight("bold")
ax.set_title("Overall Churn Rate", fontweight="bold")

# 2b. Churn by contract type
ax = axes[0, 1]
ct = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
bars = ax.bar(ct.index, ct.values, color=["#e74c3c", "#e67e22", "#27ae60"], edgecolor="white")
ax.set_title("Churn Rate by Contract Type", fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_ylim(0, ct.max() * 1.25)
for bar, val in zip(bars, ct.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
ax.tick_params(axis="x", rotation=15)

# 2c. Tenure distribution by churn
ax = axes[0, 2]
for label, color in [("Retained", "#2ecc71"), ("Churned", "#e74c3c")]:
    flag = 0 if label == "Retained" else 1
    ax.hist(df[df["Churn"] == flag]["tenure"], bins=20, alpha=0.65,
            color=color, label=label, edgecolor="white")
ax.set_title("Tenure Distribution by Churn", fontweight="bold")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Count")
ax.legend()

# 2d. Monthly charges by churn
ax = axes[1, 0]
churned = df[df["Churn"] == 1]["MonthlyCharges"]
retained = df[df["Churn"] == 0]["MonthlyCharges"]
ax.violinplot([retained, churned], positions=[1, 2], showmedians=True,
              showextrema=True)
ax.set_xticks([1, 2])
ax.set_xticklabels(["Retained", "Churned"])
ax.set_title("Monthly Charges Distribution", fontweight="bold")
ax.set_ylabel("Monthly Charges ($)")
ax.get_children()[0].set_facecolor("#2ecc71")
ax.get_children()[0].set_alpha(0.7)
ax.get_children()[2].set_facecolor("#e74c3c")
ax.get_children()[2].set_alpha(0.7)

# 2e. Internet service impact
ax = axes[1, 1]
isvc = df.groupby("InternetService")["Churn"].mean() * 100
bars = ax.bar(isvc.index, isvc.values, color=["#3498db", "#9b59b6", "#95a5a6"], edgecolor="white")
ax.set_title("Churn Rate by Internet Service", fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_ylim(0, isvc.max() * 1.25)
for bar, val in zip(bars, isvc.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

# 2f. Payment method heatmap
ax = axes[1, 2]
pm = df.groupby("PaymentMethod")["Churn"].mean() * 100
pm = pm.sort_values(ascending=True)
colors_hm = ["#27ae60" if v < 20 else "#e67e22" if v < 35 else "#e74c3c" for v in pm.values]
bars = ax.barh(pm.index, pm.values, color=colors_hm, edgecolor="white")
ax.set_title("Churn Rate by Payment Method", fontweight="bold")
ax.set_xlabel("Churn Rate (%)")
for bar, val in zip(bars, pm.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontweight="bold", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/01_eda_overview.png", bbox_inches="tight", dpi=130)
plt.close()
print("    Saved: outputs/01_eda_overview.png")

# ── 3. Feature Engineering ─────────────────────────────────────────────────────
print("\n[3] Engineering features...")

df_model = df.copy()

# Binary encode yes/no columns
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for col in binary_cols:
    df_model[col] = (df_model[col] == "Yes").astype(int)

df_model["SeniorCitizen"] = df_model["SeniorCitizen"].astype(int)

# Ordinal encode contract
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df_model["Contract"] = df_model["Contract"].map(contract_map)

# One-hot encode remaining categoricals
cat_cols = ["gender", "MultipleLines", "InternetService",
            "OnlineSecurity", "TechSupport", "PaymentMethod"]
df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
df_model.drop(columns=["customerID"], inplace=True)

# Derived features
df_model["ChargePerMonth"] = df_model["TotalCharges"] / (df_model["tenure"] + 1)
df_model["IsNewCustomer"] = (df_model["tenure"] <= 6).astype(int)
df_model["IsLoyalCustomer"] = (df_model["tenure"] >= 48).astype(int)

X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]
print(f"    Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} columns")

# ── 4. Train/Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"    Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── 5. Models ──────────────────────────────────────────────────────────────────
print("\n[4] Training models...")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=0.5),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8,
                                             min_samples_leaf=5, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                                     max_depth=4, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    X_tr = X_train_sc if name == "Logistic Regression" else X_train
    X_te = X_test_sc if name == "Logistic Regression" else X_test

    cv_scores = cross_val_score(model,
                                X_train_sc if name == "Logistic Regression" else X_train,
                                y_train, scoring="roc_auc", cv=cv)
    model.fit(X_tr, y_train)
    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred = model.predict(X_te)

    test_auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    results[name] = {
        "model": model,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_auc": test_auc,
        "avg_precision": ap,
        "X_test": X_te,
    }
    print(f"    {name:25s} | CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Test AUC: {test_auc:.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]["test_auc"])
best = results[best_name]
print(f"\n    ★ Best model: {best_name} (AUC = {best['test_auc']:.4f})")

# ── 6. Evaluation Plots ────────────────────────────────────────────────────────
print("\n[5] Generating evaluation charts...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Evaluation", fontsize=15, fontweight="bold")

colors_model = {"Logistic Regression": "#3498db", "Random Forest": "#2ecc71", "Gradient Boosting": "#e74c3c"}

# ROC curves
ax = axes[0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['test_auc']:.3f})",
            color=colors_model[name], lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.fill_between(*roc_curve(y_test, best["y_proba"])[:2],
                alpha=0.08, color=colors_model[best_name])
ax.set_title("ROC Curves", fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

# Precision-Recall curves
ax = axes[1]
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
    ap = res["avg_precision"]
    ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})",
            color=colors_model[name], lw=2)
baseline = y_test.mean()
ax.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Baseline ({baseline:.2f})")
ax.set_title("Precision-Recall Curves", fontweight="bold")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend(fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

# Confusion matrix for best model
ax = axes[2]
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Predicted Retained", "Predicted Churned"],
            yticklabels=["Actual Retained", "Actual Churned"],
            cbar=False, linewidths=0.5, linecolor="white")
ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
ax.tick_params(axis="x", rotation=20, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=9)

plt.tight_layout()
plt.savefig("outputs/02_model_evaluation.png", bbox_inches="tight", dpi=130)
plt.close()
print("    Saved: outputs/02_model_evaluation.png")

# ── 7. Feature Importance ──────────────────────────────────────────────────────
print("\n[6] Computing feature importance...")

best_model_obj = best["model"]
feat_names = X.columns.tolist()

if hasattr(best_model_obj, "feature_importances_"):
    importances = best_model_obj.feature_importances_
else:
    importances = np.abs(best_model_obj.coef_[0])

feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 7))
bar_colors = ["#e74c3c" if i < 5 else "#e67e22" if i < 10 else "#3498db"
              for i in range(len(feat_imp))]
bars = ax.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=bar_colors[::-1], edgecolor="white")
ax.set_title(f"Top 15 Churn Drivers — {best_name}", fontsize=14, fontweight="bold")
ax.set_xlabel("Feature Importance Score")
for bar, val in zip(bars, feat_imp.values[::-1]):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8)

patches = [
    mpatches.Patch(color="#e74c3c", label="High Impact"),
    mpatches.Patch(color="#e67e22", label="Medium Impact"),
    mpatches.Patch(color="#3498db", label="Lower Impact"),
]
ax.legend(handles=patches, loc="lower right")
plt.tight_layout()
plt.savefig("outputs/03_feature_importance.png", bbox_inches="tight", dpi=130)
plt.close()
print("    Saved: outputs/03_feature_importance.png")

# ── 8. Business Intelligence ───────────────────────────────────────────────────
print("\n[7] Business analysis...")

# Attach predictions to original test set
test_idx = y_test.index
X_test_orig = df.loc[test_idx].copy()
X_test_orig["ChurnProbability"] = best["y_proba"]
X_test_orig["PredictedChurn"] = best["y_pred"]
X_test_orig["RiskTier"] = pd.cut(
    X_test_orig["ChurnProbability"],
    bins=[0, 0.30, 0.55, 0.75, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
)

# Cost-benefit analysis
AVG_CLV = 1200          # Average customer lifetime value ($)
RETENTION_COST = 80     # Cost of retention offer per customer ($)
RETENTION_SUCCESS = 0.35  # Probability offer works

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Business Strategy — Churn Intervention Analysis", fontsize=14, fontweight="bold")

# Risk tier distribution
ax = axes[0]
tier_counts = X_test_orig["RiskTier"].value_counts().sort_index()
tier_colors = ["#27ae60", "#f39c12", "#e67e22", "#e74c3c"]
bars = ax.bar(tier_counts.index, tier_counts.values, color=tier_colors, edgecolor="white", width=0.6)
ax.set_title("Customer Risk Distribution", fontweight="bold")
ax.set_ylabel("Number of Customers")
ax.set_xlabel("Risk Tier")
for bar, val in zip(bars, tier_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:,}", ha="center", fontweight="bold")

# ROI at different threshold strategies
ax = axes[1]
thresholds = np.linspace(0.1, 0.9, 50)
roi_values = []
for thresh in thresholds:
    targeted = X_test_orig[X_test_orig["ChurnProbability"] >= thresh]
    true_churners = targeted[targeted["Churn"] == 1]
    cost = len(targeted) * RETENTION_COST
    revenue_saved = len(true_churners) * AVG_CLV * RETENTION_SUCCESS
    roi_values.append(revenue_saved - cost)

best_thresh_idx = np.argmax(roi_values)
best_thresh = thresholds[best_thresh_idx]
best_roi = roi_values[best_thresh_idx]

ax.plot(thresholds, roi_values, color="#3498db", lw=2.5)
ax.fill_between(thresholds, 0, roi_values,
                where=[r > 0 for r in roi_values], alpha=0.15, color="#2ecc71")
ax.fill_between(thresholds, 0, roi_values,
                where=[r <= 0 for r in roi_values], alpha=0.15, color="#e74c3c")
ax.axvline(best_thresh, color="#e74c3c", linestyle="--", lw=1.5,
           label=f"Optimal threshold = {best_thresh:.2f}")
ax.axhline(0, color="black", lw=0.8, alpha=0.5)
ax.scatter([best_thresh], [best_roi], color="#e74c3c", s=100, zorder=5)
ax.annotate(f"Max ROI\n${best_roi:,.0f}", xy=(best_thresh, best_roi),
            xytext=(best_thresh + 0.08, best_roi * 0.85),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
ax.set_title("Revenue Saved vs. Threshold Strategy", fontweight="bold")
ax.set_xlabel("Churn Probability Threshold for Intervention")
ax.set_ylabel("Net ROI ($)")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

plt.tight_layout()
plt.savefig("outputs/04_business_strategy.png", bbox_inches="tight", dpi=130)
plt.close()
print("    Saved: outputs/04_business_strategy.png")

# ── 9. High-Risk Customer Report ───────────────────────────────────────────────
high_risk = (X_test_orig[X_test_orig["ChurnProbability"] >= best_thresh]
             .sort_values("ChurnProbability", ascending=False)
             [["customerID", "Contract", "MonthlyCharges", "tenure",
               "InternetService", "ChurnProbability", "RiskTier"]]
             .head(20)
             .reset_index(drop=True))
high_risk.to_csv("outputs/high_risk_customers.csv", index=False)
print(f"    Saved: outputs/high_risk_customers.csv ({len(high_risk)} customers)")

# ── 10. Summary Report ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EXECUTIVE SUMMARY")
print("=" * 60)
print(f"  Dataset          : {len(df):,} telecom customers")
print(f"  Overall Churn    : {df['Churn'].mean():.1%}")
print(f"  Best Model       : {best_name}")
print(f"  Test ROC-AUC     : {best['test_auc']:.4f}")
print(f"  Avg Precision    : {best['avg_precision']:.4f}")
print(f"\n  KEY CHURN DRIVERS (from feature importance):")
for i, (feat, score) in enumerate(feat_imp.head(5).items(), 1):
    clean = feat.replace("_", " ").replace("Month-to-month", "M2M")
    print(f"    {i}. {clean:35s} ({score:.4f})")
print(f"\n  BUSINESS IMPACT:")
total_high_risk = len(X_test_orig[X_test_orig["ChurnProbability"] >= best_thresh])
est_churners = int(total_high_risk * X_test_orig[X_test_orig["ChurnProbability"] >= best_thresh]["Churn"].mean())
cost_of_action = total_high_risk * RETENTION_COST
revenue_at_risk = est_churners * AVG_CLV
print(f"    High-risk customers  : {total_high_risk:,}")
print(f"    Estimated true churn : {est_churners:,}")
print(f"    Revenue at risk      : ${revenue_at_risk:,.0f}")
print(f"    Intervention cost    : ${cost_of_action:,.0f}")
print(f"    Net ROI (est.)       : ${best_roi:,.0f}")
print(f"\n  Optimal intervention threshold : {best_thresh:.2f}")
print(f"\n  Output files in /outputs/")
print("=" * 60)
