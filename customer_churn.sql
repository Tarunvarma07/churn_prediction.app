SELECT * FROM machinelearning.customer_churn;
SELECT * FROM machinelearning.customer_churn LIMIT 10;
SELECT Churn, COUNT(*) AS total
FROM machinelearning.customer_churn
GROUP BY Churn;
SELECT 
    SUM(CASE WHEN TotalCharges IS NULL OR TotalCharges='' THEN 1 ELSE 0 END) AS missing_totalcharges
FROM machinelearning.customer_churn;

SELECT Contract, Churn, COUNT(*) AS total
FROM machinelearning.customer_churn
GROUP BY Contract, Churn
ORDER BY Contract;

SELECT Churn, AVG(MonthlyCharges) AS avg_monthly
FROM machinelearning.customer_churn
GROUP BY Churn;

SELECT 
    CASE 
        WHEN tenure BETWEEN 0 AND 12 THEN '0-1 year'
        WHEN tenure BETWEEN 13 AND 24 THEN '1-2 years'
        WHEN tenure BETWEEN 25 AND 60 THEN '2-5 years'
        ELSE '5+ years'
    END AS tenure_group,
    Churn, COUNT(*) AS total
FROM  machinelearning.customer_churn
GROUP BY tenure_group, Churn
ORDER BY tenure_group;


SELECT PaymentMethod, Churn, COUNT(*) AS total
FROM machinelearning.customer_churn
GROUP BY PaymentMethod, Churn
ORDER BY PaymentMethod;

SELECT MIN(MonthlyCharges), MAX(MonthlyCharges), AVG(MonthlyCharges)
FROM machinelearning.customer_churn;

