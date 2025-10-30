-- Step 5: Data Validation & EDA Roadmap
-- 1. SQL Data Validation & Sanity Checks
-- Basic Validation Queries
-- Count check for each table
SELECT COUNT(*) AS TotalCustomers FROM Customers;
SELECT COUNT(*) AS TotalLoans FROM Loans;
SELECT COUNT(*) AS TotalRepayments FROM Repayments;
SELECT COUNT(*) AS TotalSnapshots FROM LoanPerformanceSnapshots;
SELECT COUNT(*) AS TotalMacroRecords FROM Macroeconomics;

-- Foreign key consistency
SELECT COUNT(*) AS OrphanLoans
FROM Loans l
LEFT JOIN Customers c ON l.CustomerID = c.CustomerID
WHERE c.CustomerID IS NULL;

SELECT COUNT(*) AS OrphanRepayments
FROM Repayments r
LEFT JOIN Loans l ON r.LoanID = l.LoanID
WHERE l.LoanID IS NULL;

-- Check date logic
SELECT TOP 10 LoanID, OriginationDate 
FROM Loans
WHERE OriginationDate > GETDATE();

-- Range sanity checks
SELECT MIN(CreditScore) AS MinScore, MAX(CreditScore) AS MaxScore FROM Customers;
SELECT MIN(InterestRate) AS MinIR, MAX(InterestRate) AS MaxIR FROM Loans;
SELECT MIN(LoanAmount), MAX(LoanAmount) FROM Loans;

-- 2. Analytical EDA
-- Portfolio Overview
-- Total Portfolio Value
SELECT SUM(LoanAmount) AS TotalLoanAmount,
       COUNT(*) AS TotalLoans,
       AVG(InterestRate) AS AvgInterestRate
FROM Loans;

-- Status Distribution
SELECT Status, COUNT(*) AS LoanCount, 
       SUM(LoanAmount) AS TotalAmount,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM Loans), 2) AS PctOfTotal
FROM Loans
GROUP BY Status
ORDER BY LoanCount DESC;

-- Risk Grade Mix
SELECT RiskGradeAtOrigination, COUNT(*) AS LoanCount,
       AVG(InterestRate) AS AvgIR, AVG(LoanAmount) AS AvgLoan
FROM Loans
GROUP BY RiskGradeAtOrigination
ORDER BY RiskGradeAtOrigination;

-- Customer Insights
-- Average loan per industry
SELECT c.Industry,
       COUNT(l.LoanID) AS NumLoans,
       AVG(l.LoanAmount) AS AvgLoan,
       SUM(l.LoanAmount) AS TotalDisbursed,
       AVG(c.CreditScore) AS AvgCreditScore
FROM Customers c
JOIN Loans l ON c.CustomerID = l.CustomerID
GROUP BY c.Industry
ORDER BY TotalDisbursed DESC;

-- Region-wise default rate
SELECT c.Region,
       SUM(CASE WHEN l.Status = 'Defaulted' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS DefaultRate,
       COUNT(*) AS NumLoans
FROM Loans l
JOIN Customers c ON c.CustomerID = l.CustomerID
GROUP BY c.Region
ORDER BY DefaultRate DESC;

-- Repayment Behavior
-- Average delay in repayments by risk grade
SELECT l.RiskGradeAtOrigination,
       AVG(r.DaysLate) AS AvgDelay,
       COUNT(DISTINCT l.LoanID) AS NumLoans
FROM Repayments r
JOIN Loans l ON r.LoanID = l.LoanID
GROUP BY l.RiskGradeAtOrigination
ORDER BY AvgDelay DESC;

-- Paid amount ratio
SELECT l.Status,
       SUM(r.PaymentAmount) / SUM(r.DueAmount) AS PaymentRatio
FROM Repayments r
JOIN Loans l ON r.LoanID = l.LoanID
GROUP BY l.Status;