# Scenario and Stress Testing

## 1. Objective
To evaluate the sensitivity of predicted default rates under different macroeconomic and portfolio conditions.

---

## 2. Scenarios Defined

| Scenario | Inflation | GDP Growth | Interest Rate | Description |
|-----------|------------|-------------|----------------|--------------|
| Base Case | 5% | 6.2% | 6.5% | Normal growth environment |
| Mild Stress | 7% | 4% | 7.5% | Moderate inflation shock |
| Severe Stress | 9% | 2% | 8.5% | Recessionary environment |
| Recovery | 4% | 7% | 6% | Economic rebound scenario |

---

## 3. Methodology
- Adjusted macro variables (`InflationRate`, `GDPGrowthRate`, `CentralBankInterestRate`) within test dataset.  
- Recomputed predicted probabilities using the trained model.  
- Compared portfolio-level Expected Loss (EL) under each scenario:

  **Expected Loss (EL) = PD × LGD × Exposure**

---

## 4. Results Summary

| Scenario | Mean PD | Expected Loss (₹ millions) | Δ vs Base |
|-----------|----------|-----------------------------|-----------|
| Base Case | 0.086 | 125.4 | — |
| Mild Stress | 0.102 | 148.9 | +18.8% |
| Severe Stress | 0.132 | 195.7 | +56.0% |
| Recovery | 0.071 | 102.6 | -18.2% |

---

## 5. Insights
- Portfolio default risk increases sharply under inflationary and low-GDP conditions.  
- Exposure-weighted loss under **Severe Stress** could rise by ~56%.  
- Industry exposure to construction and retail sectors amplifies downturn sensitivity.  
- Recovery scenarios show potential for a 15–20% reduction in loss expectation.

---

## 6. Recommendations
- Introduce dynamic loan pricing tied to predicted PD and macro conditions.  
- Maintain higher capital buffers for cyclical sectors.  
- Prioritize proactive restructuring for high-PD clients under stress.
