# Methodology and Model Design

## 1. Objective
To predict the probability of default (PD) for SME loans and support risk-based portfolio management.

## 2. Data Pipeline Design
Data sourced from Microsoft SQL Server containing:
- **Loans**: Loan details, interest, term, origination date  
- **Customers**: Firm demographics, credit score, revenue  
- **Repayments**: Transactional repayment records  
- **Snapshots**: Point-in-time performance data  
- **Macroeconomics**: Inflation, GDP growth, central bank rates

All datasets were joined to produce one analytical base table (one row per loan).

## 3. Feature Engineering
Derived new variables to capture key behavioral and financial patterns:
- **LoanToRevenue** = LoanAmount / AnnualRevenue  
- **PaymentCoverage** = TotalPaid / LoanAmount  
- **RemainingRatio** = OutstandingAmount / LoanAmount  
- **LoanAgeMonths** = Months since origination  
- **MonthsSinceLastPayment** = LoanAgeMonths – NumPayments  
- Categorical encoding: Industry, Region, RiskGrade, CreditScoreBand

## 4. Model Selection
- Baseline: Logistic Regression (class_weight='balanced')  
- Advanced: Random Forest (RandomizedSearchCV optimization)  
- Handled imbalance using **SMOTE** (raised minority class to 30%).  
- Evaluation Metrics: ROC-AUC, PR-AUC, F1-score, and calibration.

## 5. Validation
- Stratified 80/20 train-test split  
- 3-fold cross-validation for hyperparameter tuning  
- Calibration curve and threshold optimization based on business objectives (F1-score maximization)

## 6. Explainability
- **Feature Importances**: Identified top predictors influencing default  
- **SHAP Analysis**: Provided interpretability and direction of feature impact  
- **Dependence Plots**: Visualized marginal contribution of high-impact variables

## 7. Deployment & Scoring
- Model exported via `joblib`  
- Predictions written back to SQL table `PredictedLoanScores`  
- Output CSV generated for Power BI integration

## 8. Limitations
- Historical data limited to available repayment and macroeconomic snapshots  
- No behavioral data post-origination beyond loan age considered  
- Static model (not yet dynamic retraining or temporal validation)

## 9. Tools Used
- **SQL Server** – Data preparation  
- **Python (pandas, scikit-learn, shap, imblearn)** – Modeling and analysis  
- **Power BI** – Visualization and business interpretation  
