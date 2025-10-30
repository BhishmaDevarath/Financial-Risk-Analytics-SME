# --- Cell 1 — Install requirements ---
# Run this cell in your environment (only if packages not installed)
# In VS Code terminal you could run: pip install -r requirements.txt
# Or run these individually (uncomment then run)

# !pip install pandas numpy scikit-learn matplotlib seaborn pyodbc joblib shap imbalanced-learn xgboost

# Note: SHAP can be heavy; install it if you plan to run explanation steps.

# --- Cell 2 — Imports & config ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sqlalchemy import create_engine
from sqlalchemy import text
from datetime import datetime

# sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# imblearn
from imblearn.over_sampling import SMOTE

# SHAP (optional)
import shap

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set(style="whitegrid")

# --- Cell 3 — DB connection helper ---
server = 'localhost'  # or 'localhost\\SQLEXPRESS' if you use the SQLEXPRESS instance
database = 'SME_Loan_Portfolio'

# Build SQLAlchemy connection string for Windows Authentication
driver = 'ODBC Driver 17 for SQL Server'
server = 'localhost'  # or 'localhost\\SQLEXPRESS'
database = 'SME_Loan_Portfolio'

connection_url = f"mssql+pyodbc://@{server}/{database}?driver={driver.replace(' ', '+')}&trusted_connection=yes"

# Create SQLAlchemy engine
engine = create_engine(connection_url)
print("✅ Connected to SQL Server successfully via SQLAlchemy (Windows Authentication)")

'''
# --- Cell 4 — SQL: build modeling dataset (one row per loan) ---
-- (This SQL is for reference; you can run it in SQL Server to create a modeling view or run from Python)
-- It aggregates repayment behavior and joins customer + macro data.
-- Save as a view called vw_LoanModelDataset if desired.

-- Example SQL (run in SSMS or translate to Python queries):

/*
CREATE VIEW vw_LoanModelDataset AS
SELECT
  l.LoanID,
  l.CustomerID,
  l.LoanAmount,
  l.InterestRate,
  l.TermMonths,
  l.OriginationDate,
  l.Status,
  l.RiskGradeAtOrigination,
  c.Industry,
  c.Region,
  c.YearsInBusiness,
  c.AnnualRevenue,
  c.CreditScore,
  -- repayment aggregates
  ISNULL(rp.NumPayments, 0) AS NumPayments,
  ISNULL(rp.AvgDaysLate, 0) AS AvgDaysLate,
  ISNULL(rp.TotalPaid, 0) AS TotalPaid,
  ISNULL(rp.PaymentRatio, 0) AS PaymentRatio,
  -- snapshot fields
  s.OutstandingAmount,
  s.PastDueDays,
  s.CumulativePayments,
  s.DefaultFlag,
  -- macro: join on month of origination (nearest)
  m.InflationRate,
  m.GDPGrowthRate,
  m.CentralBankInterestRate
FROM Loans l
LEFT JOIN Customers c ON l.CustomerID = c.CustomerID
LEFT JOIN (
    SELECT LoanID,
           COUNT(*) AS NumPayments,
           AVG(DaysLate) AS AvgDaysLate,
           SUM(PaymentAmount) AS TotalPaid,
           CASE WHEN SUM(DueAmount)=0 THEN 0 ELSE SUM(PaymentAmount)*1.0/SUM(DueAmount) END AS PaymentRatio
    FROM Repayments
    GROUP BY LoanID
) rp ON rp.LoanID = l.LoanID
LEFT JOIN LoanPerformanceSnapshots s ON s.LoanID = l.LoanID
LEFT JOIN Macroeconomics m ON FORMAT(l.OriginationDate,'yyyy-MM-01') = FORMAT(m.MonthDate,'yyyy-MM-01');
*/
'''

# --- Cell 5 — Pull modeling data from SQL into pandas ---
# Pull loans + customers + repayments aggregates + snapshot + macro via queries
# 1) Load loans and customers
loans = pd.read_sql("SELECT * FROM Loans", engine)
customers = pd.read_sql("SELECT * FROM Customers", engine)
snapshots = pd.read_sql("SELECT * FROM LoanPerformanceSnapshots", engine)
macros = pd.read_sql("SELECT * FROM Macroeconomics", engine)

# 2) Repayments aggregates
repayment_agg_sql = """
SELECT 
    LoanID,
    COUNT(*) AS NumPayments,
    AVG(DaysLate) AS AvgDaysLate,
    SUM(PaymentAmount) AS TotalPaid,
    CASE WHEN SUM(DueAmount)=0 THEN 0 ELSE SUM(PaymentAmount)*1.0/SUM(DueAmount) END AS PaymentRatio
FROM Repayments
GROUP BY LoanID
"""
rep_agg = pd.read_sql(repayment_agg_sql, engine)

# 3) Merge all
# Convert types
loans['OriginationDate'] = pd.to_datetime(loans['OriginationDate'])
snapshots['SnapshotDate'] = pd.to_datetime(snapshots['SnapshotDate'])
macros['MonthDate'] = pd.to_datetime(macros['MonthDate'])

# Merge loans->customers->rep_agg->snapshots->macros (left joins)
df = loans.merge(customers, on='CustomerID', how='left') \
          .merge(rep_agg, on='LoanID', how='left') \
          .merge(snapshots[['LoanID','OutstandingAmount','PastDueDays','CumulativePayments','DefaultFlag']], on='LoanID', how='left')

# Join macro by matching year-month of origination to monthdate
df['OriginationMonth'] = df['OriginationDate'].dt.to_period('M').dt.to_timestamp()
macros['MonthMonth'] = macros['MonthDate'].dt.to_period('M').dt.to_timestamp()
df = df.merge(macros[['MonthMonth','InflationRate','GDPGrowthRate','CentralBankInterestRate']], left_on='OriginationMonth', right_on='MonthMonth', how='left')
df.drop(columns=['MonthMonth'], inplace=True)

# Fill NaNs from aggregates with 0 or reasonable defaults
df['NumPayments'] = df['NumPayments'].fillna(0).astype(int)
df['AvgDaysLate'] = df['AvgDaysLate'].fillna(0)
df['TotalPaid'] = df['TotalPaid'].fillna(0.0)
df['PaymentRatio'] = df['PaymentRatio'].fillna(0.0)
df['OutstandingAmount'] = df['OutstandingAmount'].fillna(df['LoanAmount'])  # if no snapshot, outstanding ~ loan amount
df['PastDueDays'] = df['PastDueDays'].fillna(0)
df['CumulativePayments'] = df['CumulativePayments'].fillna(0.0)
df['DefaultFlag'] = df['DefaultFlag'].fillna(0).astype(int)

print("Modeling dataframe shape:", df.shape)
df.head()

# --- Cell 6 — Feature engineering ---
# Create features that are predictive and business-meaningful

# Basic ratio features
df['LoanToRevenue'] = df['LoanAmount'] / (df['AnnualRevenue'] + 1)   # +1 to avoid div by zero
df['PaymentCoverage'] = df['TotalPaid'] / (df['LoanAmount'] + 1)
df['RemainingRatio'] = df['OutstandingAmount'] / (df['LoanAmount'] + 1)

# Recency / age features
today = pd.Timestamp(datetime.now().date())
df['LoanAgeMonths'] = ((today - df['OriginationDate']).dt.days / 30).astype(int)
df['MonthsSinceLastPayment'] = np.maximum(0, df['LoanAgeMonths'] - df['NumPayments'])

# Binning / groups
df['CreditScoreBand'] = pd.cut(df['CreditScore'], bins=[299, 600, 660, 720, 900], labels=['Low','Med','High','VeryHigh'])

# One-hot encode categorical features (industry, region, riskgrade) - keep limited to top categories
cat_cols = ['Industry','Region','RiskGradeAtOrigination','CreditScoreBand']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Select final feature set
feature_cols = [
    'LoanAmount','InterestRate','TermMonths','YearsInBusiness','AnnualRevenue','CreditScore',
    'NumPayments','AvgDaysLate','PaymentRatio','OutstandingAmount','PastDueDays','CumulativePayments',
    'LoanToRevenue','PaymentCoverage','RemainingRatio','LoanAgeMonths','MonthsSinceLastPayment',
    'InflationRate','GDPGrowthRate','CentralBankInterestRate'
]
# add any encoded categorical columns
encoded_cols = [c for c in df.columns if any(prefix in c for prefix in ['Industry_','Region_','RiskGradeAtOrigination_','CreditScoreBand_'])]
feature_cols += encoded_cols

# Ensure all features exist
feature_cols = [c for c in feature_cols if c in df.columns]
len(feature_cols), feature_cols[:10]

# --- Cell 7 — Train/test split & scaling ---
X = df[feature_cols].copy()
y = df['DefaultFlag'].copy()

# Quick class balance
print("Default rate:", y.mean())

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# Scaling numerical features when needed - create pipeline later
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# --- Cell 8 — Baseline model: Logistic Regression (class weight) ---
# Baseline logistic regression with class_weight='balanced' to handle imbalance
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)

# Predictions & metrics
y_pred_proba = lr.predict_proba(X_test_scaled)[:,1]
y_pred = (y_pred_proba >= 0.5).astype(int)

print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print(classification_report(y_test, y_pred, digits=4))
print("Average precision (PR AUC):", average_precision_score(y_test, y_pred_proba))

# --- Cell 9 — Stronger model: Random Forest + CV + Randomized Search ---
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced')

# We'll perform a randomized search for a few hyperparams
param_dist = {
    'n_estimators': [100,200,400],
    'max_depth': [6,10,15, None],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4],
    'max_features': ['sqrt','log2', None]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, scoring='roc_auc', n_jobs=-1, cv=cv, random_state=RANDOM_STATE, verbose=1)
rand_search.fit(X_train, y_train)   # random forest works on unscaled features

print("Best params:", rand_search.best_params_)
best_rf = rand_search.best_estimator_

# Evaluate on test set
y_proba_rf = best_rf.predict_proba(X_test)[:,1]
print("RF ROC AUC:", roc_auc_score(y_test, y_proba_rf))
print("RF Avg Precision:", average_precision_score(y_test, y_proba_rf))

# --- Cell 10 — Handle imbalance with SMOTE and re-train ---
# SMOTE oversampling on training set
sm = SMOTE(sampling_strategy=0.3, random_state=RANDOM_STATE)  # raise minority to 30% of majority
X_res, y_res = sm.fit_resample(X_train, y_train)
print("After SMOTE:", np.bincount(y_res))

rf_sm = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf_sm.fit(X_res, y_res)
y_proba_rf_sm = rf_sm.predict_proba(X_test)[:,1]
print("RF (SMOTE) ROC AUC:", roc_auc_score(y_test, y_proba_rf_sm))
print("RF (SMOTE) Avg Precision:", average_precision_score(y_test, y_proba_rf_sm))

# --- Cell 11 — Calibration & threshold selection (business metric) ---
# Calibration curve for best model (use best_rf or rf_sm)
probas = y_proba_rf  # choose the model you prefer
prob_true, prob_pred = calibration_curve(y_test, probas, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1], linestyle='--')
plt.title('Calibration plot')
plt.xlabel('Predicted probability')
plt.ylabel('Observed frequency')
plt.show()

# Choose threshold using Precision-Recall tradeoff or business cost (example)
precision, recall, thresholds = precision_recall_curve(y_test, probas)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
print("Best threshold by F1:", best_threshold, "Precision:", precision[best_idx], "Recall:", recall[best_idx])

# --- Cell 12 — Confusion matrix & business KPIs ---
# Apply threshold
threshold = best_threshold
y_pred_thresh = (probas >= threshold).astype(int)

cm = confusion_matrix(y_test, y_pred_thresh)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (threshold={threshold:.2f})')
plt.show()

print(classification_report(y_test, y_pred_thresh, digits=4))

# Example expected loss calculation (simplified)
# Assume LGD (loss given default) = 0.6 and Exposure = OutstandingAmount in snapshot
lgd = 0.6
sample = df.loc[y_test.index]  # align test index with df
sample = sample.copy()
sample['pred_proba'] = probas
sample['pred_flag'] = y_pred_thresh
# expected_loss = sum(pred_prob * Exposure * LGD)
expected_loss = (sample['pred_proba'] * sample['OutstandingAmount'] * lgd).sum()
print("Estimated portfolio expected loss (test set):", expected_loss)

# --- Cell 13 — Feature importance & SHAP explanations (FIXED) ---
# Feature importance (RandomForest)
importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(x=importances.values[:20], y=importances.index[:20])
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

# Ensure only numeric features are passed to SHAP
X_train_shap = X_train.select_dtypes(include=[np.number]).copy()

# SHAP (if installed) for explainability
# Use a smaller sample for faster computation
sample_size = min(1000, len(X_train_shap))
X_train_shap_sample = X_train_shap.sample(n=sample_size, random_state=RANDOM_STATE)

try:
    explainer = shap.Explainer(best_rf, X_train_shap_sample)
    shap_values = explainer(X_train_shap_sample, check_additivity=False)
    
    # Summary plot (suppress the numpy warning)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        shap.summary_plot(shap_values, X_train_shap_sample, show=True)
    
    # Create manual dependence plot instead of using shap.dependence_plot
    # (shap.dependence_plot has known issues with some data shapes)
    top_feat = importances.index[0]
    if top_feat in X_train_shap_sample.columns:
        try:
            plt.figure(figsize=(10, 6))
            feat_idx = list(X_train_shap_sample.columns).index(top_feat)
            
            # Create scatter plot with color based on feature value
            scatter = plt.scatter(
                X_train_shap_sample[top_feat],
                shap_values.values[:, feat_idx],
                c=X_train_shap_sample[top_feat],
                cmap='viridis',
                alpha=0.6,
                s=20
            )
            plt.colorbar(scatter, label=top_feat)
            plt.xlabel(f'{top_feat} (feature value)')
            plt.ylabel(f'SHAP value for {top_feat}')
            plt.title(f'SHAP Dependence Plot: {top_feat}')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()
            print(f"✅ Created SHAP dependence plot for {top_feat}")
        except Exception as e:
            print(f"⚠️  Could not create dependence plot for {top_feat}: {e}")
    else:
        print(f"Top feature {top_feat} not found in numeric columns")
except Exception as e:
    print(f"⚠️  SHAP analysis failed: {e}")
    print("Continuing without SHAP plots...")

# --- Cell 14 — Save the model & scaler ---
os.makedirs('../models', exist_ok=True)
joblib.dump(best_rf, '../models/rf_default_model.joblib')
joblib.dump(scaler, '../models/feature_scaler.joblib')
print("Saved model to ../models/")

# --- Cell 15 — Scoring new loans and writing scores back to SQL ---
# Example: score all loans in DB and write predictions to table PredictedLoanScores
# Build X_all (same feature engineering applied to full df)
X_all = df[feature_cols].copy()
# if you scaled features for LR pipeline: apply scaler to num columns if needed
# For RF we used raw features; ensure pipeline consistent.

# Use saved model
model = best_rf
probas_all = model.predict_proba(X_all)[:,1]
df_scores = df[['LoanID','CustomerID','LoanAmount','OutstandingAmount']].copy()
df_scores['pred_proba'] = probas_all
df_scores['pred_flag'] = (df_scores['pred_proba'] >= threshold).astype(int)

# Create table in SQL for predicted scores and upsert/insert (example)
create_table_sql = """
IF OBJECT_ID('dbo.PredictedLoanScores', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.PredictedLoanScores (
        LoanID INT PRIMARY KEY,
        CustomerID INT,
        LoanAmount DECIMAL(18,2),
        OutstandingAmount DECIMAL(18,2),
        PredProb FLOAT,
        PredFlag BIT,
        ScoreDate DATETIME DEFAULT GETDATE()
    );
END
"""
with engine.begin() as connection:
    connection.execute(text(create_table_sql))

# Upsert (merge) predictions in batches
merge_sql = text("""
MERGE dbo.PredictedLoanScores AS tgt
USING (VALUES (:LoanID, :CustomerID, :LoanAmount, :OutstandingAmount, :PredProb, :PredFlag))
       AS src (LoanID, CustomerID, LoanAmount, OutstandingAmount, PredProb, PredFlag)
ON tgt.LoanID = src.LoanID
WHEN MATCHED THEN 
    UPDATE SET CustomerID = src.CustomerID,
               LoanAmount = src.LoanAmount,
               OutstandingAmount = src.OutstandingAmount,
               PredProb = src.PredProb,
               PredFlag = src.PredFlag,
               ScoreDate = GETDATE()
WHEN NOT MATCHED THEN 
    INSERT (LoanID, CustomerID, LoanAmount, OutstandingAmount, PredProb, PredFlag)
    VALUES (src.LoanID, src.CustomerID, src.LoanAmount, src.OutstandingAmount, src.PredProb, src.PredFlag);
""")

with engine.begin() as connection:
    for _, row in df_scores.iterrows():
        connection.execute(merge_sql, {
            'LoanID': int(row.LoanID),
            'CustomerID': int(row.CustomerID),
            'LoanAmount': float(row.LoanAmount),
            'OutstandingAmount': float(row.OutstandingAmount),
            'PredProb': float(row.pred_proba),
            'PredFlag': int(row.pred_flag)
        })

print("✅ Scored loans and upserted PredictedLoanScores table via SQLAlchemy.")

# ------------------ Export predictions & eval for Power BI ------------------
# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Add actual label (DefaultFlag) and some customer fields to df_scores for Power BI
# df already has DefaultFlag and customer fields because df = loans.merge(customers...) earlier
# Make sure df_scores aligns to df

# Merge with Customers to get Region and Industry
cust_df = pd.read_sql("SELECT CustomerID, Region, Industry FROM Customers", engine)
df = df.merge(cust_df, on='CustomerID', how='left')

df_scores = df[['LoanID','CustomerID','LoanAmount','OutstandingAmount','DefaultFlag','Region','Industry']].copy()
df_scores['PredProb'] = probas_all
df_scores['PredFlag'] = (df_scores['PredProb'] >= threshold).astype(int)

# Save CSV for Power BI import
pred_csv_path = os.path.join('data','model_predictions.csv')
df_scores.to_csv(pred_csv_path, index=False)
print(f"Saved model predictions CSV to: {pred_csv_path}")

# Save a small metrics file for KPI cards (one row)
from sklearn.metrics import precision_score, recall_score, accuracy_score
y_test_pred = (y_proba_rf >= threshold).astype(int)
precision = precision_score(y_test, y_test_pred, zero_division=0)
recall = recall_score(y_test, y_test_pred, zero_division=0)
accuracy = accuracy_score(y_test, y_pred)  # y_pred from logistic baseline or use RF predictions if preferred
auc = roc_auc_score(y_test, y_proba_rf)

metrics_df = pd.DataFrame([{
    'Model': 'RandomForest_tuned',
    'ROC_AUC': auc,
    'Precision': precision,
    'Recall': recall,
    'Accuracy': accuracy,
    'Threshold': float(threshold),
    'TestSize': len(y_test)
}])

metrics_csv_path = os.path.join('reports','model_eval_summary.csv')
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Saved model metrics CSV to: {metrics_csv_path}")
# ---------------------------------------------------------------------------

# --- Cell 16 — Save evaluation summary for README / resume ---
# Metrics summary
eval_summary = {
    'Model': 'RandomForest (tuned)',
    'ROC_AUC': roc_auc_score(y_test, y_proba_rf),
    'AvgPrecision_PR_AUC': average_precision_score(y_test, y_proba_rf),
    'DefaultRate': y.mean(),
    'TestSize': len(y_test),
    'ThresholdUsed': threshold
}
pd.Series(eval_summary)
# Save to JSON or markdown for README
import json
os.makedirs('../reports', exist_ok=True)
with open('../reports/model_eval_summary.json','w') as f:
    json.dump(eval_summary, f, indent=2)
print("Saved model evaluation summary to ../reports/")

# --- Cleanup temporary joblib folders to prevent resource_tracker warnings ---
import tempfile, shutil

temp_dir = tempfile.gettempdir()
for folder in os.listdir(temp_dir):
    if "joblib_memmapping_folder" in folder:
        try:
            shutil.rmtree(os.path.join(temp_dir, folder), ignore_errors=True)
        except:
            pass

print("🧹 Cleaned up temporary joblib folders.")
print("\n✅ Script completed successfully!")