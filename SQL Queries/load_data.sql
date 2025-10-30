-- Step 4: Import Data into the Tables
-- 1. Import Customers Dataset
BULK INSERT Customers
FROM 'D:\SME Loan Portfolio Analytics\Data\customers.csv'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,
    FIELDQUOTE = '"',
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '0x0a',
    TABLOCK
);

-- 2. Import Loans Dataset
BULK INSERT Loans
FROM 'D:\SME Loan Portfolio Analytics\Data\loans.csv'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,
    FIELDQUOTE = '"',
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '0x0a',
    TABLOCK
);

-- 3. Import Repayments Dataset
BULK INSERT Repayments
FROM 'D:\SME Loan Portfolio Analytics\Data\repayments.csv'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,
    FIELDQUOTE = '"',
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '0x0a',
    TABLOCK
);

-- 4. Import Macroeconomics Dataset
BULK INSERT Macroeconomics
FROM 'D:\SME Loan Portfolio Analytics\Data\macroeconomics.csv'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,
    FIELDQUOTE = '"',
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '0x0a',
    TABLOCK
);

-- 5. Import LoanPerformanceSnapshots Dataset
BULK INSERT LoanPerformanceSnapshots
FROM 'D:\SME Loan Portfolio Analytics\Data\loan_performance_snapshots.csv'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,
    FIELDQUOTE = '"',
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '0x0a',
    TABLOCK
);