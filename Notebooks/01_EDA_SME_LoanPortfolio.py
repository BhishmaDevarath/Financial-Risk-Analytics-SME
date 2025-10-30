# --- Step 1: Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# --- Step 2: Database Connection ---
server = 'localhost'  # or 'localhost\\SQLEXPRESS'
database = 'SME_Loan_Portfolio'
driver = 'ODBC Driver 17 for SQL Server'

# For Windows Authentication:
conn_str = f'mssql+pyodbc://@{server}/{database}?driver={driver.replace(" ", "+")}&trusted_connection=yes'

# Create SQLAlchemy engine
engine = create_engine(conn_str)

print("✅ Connected to SQL Server via SQLAlchemy")

# --- Step 3: Load Data from SQL---
# Load all tables
customers = pd.read_sql("SELECT * FROM Customers", engine)
loans = pd.read_sql("SELECT * FROM Loans", engine)
repayments = pd.read_sql("SELECT * FROM Repayments", engine)
macros = pd.read_sql("SELECT * FROM Macroeconomics", engine)
snapshots = pd.read_sql("SELECT * FROM LoanPerformanceSnapshots", engine)

# Preview
print(customers.shape, loans.shape, repayments.shape, macros.shape, snapshots.shape)

# --- Step 4: Basic Cleaning & Type Conversions ---
# Convert dates
for df, cols in [(loans, ['OriginationDate']),
                 (repayments, ['PaymentDate']),
                 (snapshots, ['SnapshotDate']),
                 (macros, ['MonthDate'])]:
    for c in cols:
        df[c] = pd.to_datetime(df[c])

# Ensure numeric columns
numeric_cols = ['LoanAmount', 'InterestRate', 'TermMonths']
loans[numeric_cols] = loans[numeric_cols].apply(pd.to_numeric)

# Merge for richer analysis
loan_cust = loans.merge(customers, on='CustomerID', how='left')

# --- Step 5: Portfolio Overview ---
portfolio_summary = loan_cust.groupby('Status').agg(
    LoanCount=('LoanID', 'count'),
    TotalLoanAmount=('LoanAmount', 'sum'),
    AvgInterestRate=('InterestRate', 'mean')
).reset_index()

portfolio_summary['PctOfTotal'] = 100 * portfolio_summary['LoanCount'] / portfolio_summary['LoanCount'].sum()
portfolio_summary

# --- Visualization ---
sns.barplot(data=portfolio_summary, x='Status', y='LoanCount', palette='crest')
plt.title('Loan Status Distribution')
plt.show()

sns.barplot(data=portfolio_summary, x='Status', y='TotalLoanAmount', palette='flare')
plt.title('Total Loan Amount by Status')
plt.show()

# --- Step 6: Risk & Industry Analysis ---
# Risk vs Default Rate
risk_default = loan_cust.groupby('RiskGradeAtOrigination').agg(
    DefaultRate=('Status', lambda x: (x == 'Defaulted').mean()),
    AvgInterest=('InterestRate', 'mean'),
    AvgLoan=('LoanAmount', 'mean')
).reset_index()

sns.barplot(data=risk_default, x='RiskGradeAtOrigination', y='DefaultRate', palette='rocket')
plt.title('Default Rate by Risk Grade')
plt.show()

# Industry Portfolio
industry_summary = loan_cust.groupby('Industry').agg(
    TotalLoans=('LoanID', 'count'),
    AvgLoan=('LoanAmount', 'mean'),
    DefaultRate=('Status', lambda x: (x == 'Defaulted').mean())
).sort_values('TotalLoans', ascending=False)

industry_summary.plot(kind='bar', y='DefaultRate', figsize=(10,5), legend=False)
plt.title('Default Rate by Industry')
plt.show()

# --- Step 7: Regional Behavior ---
region_perf = loan_cust.groupby('Region').agg(
    NumLoans=('LoanID', 'count'),
    DefaultRate=('Status', lambda x: (x == 'Defaulted').mean()),
    AvgInterestRate=('InterestRate', 'mean')
).reset_index()

sns.barplot(data=region_perf, x='Region', y='DefaultRate', palette='coolwarm')
plt.title('Default Rate by Region')
plt.show()

# --- Step 8: Temporal & Snapshot Trends ---
snapshots['Month'] = snapshots['SnapshotDate'].dt.to_period('M').astype(str)
trend = snapshots.groupby('Month').agg(
    AvgOutstanding=('OutstandingAmount', 'mean'),
    DefaultCount=('DefaultFlag', 'sum')
).reset_index()

fig, ax1 = plt.subplots(figsize=(10,5))
sns.lineplot(data=trend, x='Month', y='AvgOutstanding', ax=ax1, label='Avg Outstanding', color='blue')
ax2 = ax1.twinx()
sns.lineplot(data=trend, x='Month', y='DefaultCount', ax=ax2, label='Defaults', color='red')
plt.title('Portfolio Trend: Outstanding vs Defaults')
plt.show()

# --- Step 9: Correlation Analysis ---
corr_cols = ['LoanAmount', 'InterestRate', 'CreditScore', 'YearsInBusiness', 'AnnualRevenue']
corr = loan_cust[corr_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()
