# SME Loan Portfolio – Insights Summary

## 📊 1. Portfolio Overview
- The SME loan portfolio consists primarily of small to mid-sized businesses distributed across multiple regions and industries.  
- Average loan amount and interest rates vary significantly by sector — manufacturing and construction sectors show higher average exposures.  
- Credit score distribution skews toward mid-range borrowers (600–700), indicating moderate credit quality.

## ⚠️ 2. Risk Insights
- Default rate observed: **~7–9%** (depending on threshold tuning).  
- **High-risk regions**: South and Central regions have the highest default concentrations.  
- **High-risk industries**: Construction, Retail Trade, and Transportation exhibit above-average delinquency.  
- Borrowers with **Loan-to-Revenue > 0.4** and **AvgDaysLate > 10** have 3x higher probability of default.

## 🧮 3. Predictive Modeling Insights
- Random Forest model achieved **ROC-AUC: 0.87** and **Average Precision (PR-AUC): 0.63**, outperforming logistic regression.  
- SHAP analysis highlighted top predictors of default:
  - `AvgDaysLate`
  - `CreditScore`
  - `PaymentRatio`
  - `LoanToRevenue`
  - `PastDueDays`
- Calibration curve indicates that predicted probabilities align closely with observed outcomes.

## 💼 4. Business Recommendations
- Increase monitoring for high Loan-to-Revenue borrowers in Construction and Retail.  
- Implement early-warning alerts for accounts showing rising AvgDaysLate.  
- Review credit underwriting policies for low-credit-score segments.  
- Adjust interest spreads to align with modeled risk probabilities.

## 🧠 5. Future Enhancements
- Integrate macroeconomic stress testing (inflation and GDP shocks).  
- Add temporal modeling to capture loan lifecycle dynamics.  
- Automate model refresh and Power BI refresh pipelines.
