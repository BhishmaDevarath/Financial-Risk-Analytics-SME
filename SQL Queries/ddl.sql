-- Step 1: Create Database

CREATE DATABASE SME_Loan_Portfolio;
GO

USE SME_Loan_Portfolio;
GO

-- Step 2: Create Tables

-- 1. Customers Table
CREATE TABLE Customers (
    CustomerID INT IDENTITY(1000,1) PRIMARY KEY,
    CustomerName NVARCHAR(100) NOT NULL,
    Industry NVARCHAR(50),
    Region NVARCHAR(50),
    YearsInBusiness INT CHECK (YearsInBusiness >= 0),
    AnnualRevenue DECIMAL(18,2),
    CreditScore INT CHECK (CreditScore BETWEEN 300 AND 900),
    CreatedAt DATETIME DEFAULT GETDATE()
);

-- 2. Loans Table
CREATE TABLE Loans (
    LoanID INT IDENTITY(5000,1) PRIMARY KEY,
    CustomerID INT NOT NULL,
    LoanAmount DECIMAL(18,2) CHECK (LoanAmount > 0),
    InterestRate DECIMAL(5,2) CHECK (InterestRate > 0),
    TermMonths INT CHECK (TermMonths > 0),
    OriginationDate DATE,
    Status NVARCHAR(30) CHECK (Status IN ('Current', 'Delinquent', 'Defaulted', 'PaidOff')),
    RiskGradeAtOrigination NVARCHAR(10),
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);

-- 3. Repayments Table
CREATE TABLE Repayments (
    RepaymentID INT IDENTITY(1,1) PRIMARY KEY,
    LoanID INT NOT NULL,
    PaymentDate DATE NOT NULL,
    PaymentAmount DECIMAL(18,2) CHECK (PaymentAmount >= 0),
    DueAmount DECIMAL(18,2) CHECK (DueAmount >= 0),
    DaysLate INT CHECK (DaysLate >= 0),
    FOREIGN KEY (LoanID) REFERENCES Loans(LoanID)
);

-- 4. Macroeconomics Table
CREATE TABLE Macroeconomics (
    MonthDate DATE PRIMARY KEY,
    InflationRate DECIMAL(5,2),
    GDPGrowthRate DECIMAL(5,2),
    CentralBankInterestRate DECIMAL(5,2)
);

-- 5. LoanPerformanceSnapshots Table
CREATE TABLE LoanPerformanceSnapshots (
    SnapshotID INT IDENTITY(1,1) PRIMARY KEY,
    LoanID INT NOT NULL,
    SnapshotDate DATE NOT NULL,
    OutstandingAmount DECIMAL(18,2),
    PastDueDays INT,
    CumulativePayments DECIMAL(18,2),
    DefaultFlag BIT,
    FOREIGN KEY (LoanID) REFERENCES Loans(LoanID)
);

-- Step 3: Create Helpful Indexes
CREATE INDEX idx_Loans_CustomerID ON Loans(CustomerID);
CREATE INDEX idx_Repayments_LoanID ON Repayments(LoanID);
CREATE INDEX idx_Loans_Status ON Loans(Status);
CREATE INDEX idx_Customers_Industry ON Customers(Industry);
CREATE INDEX idx_Customers_Region ON Customers(Region);