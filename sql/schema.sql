-- ============================================================
-- CREDIT CARD FRAUD DETECTION — SQL Schema & Queries
-- ============================================================

CREATE TABLE transactions (
    transaction_id  BIGSERIAL PRIMARY KEY,
    transaction_time FLOAT,          -- seconds from first transaction
    v1 FLOAT, v2 FLOAT, v3 FLOAT,   -- PCA components (anonymized)
    v4 FLOAT, v5 FLOAT, v6 FLOAT,
    v7 FLOAT, v8 FLOAT, v9 FLOAT,
    v10 FLOAT, v11 FLOAT, v12 FLOAT,
    v13 FLOAT, v14 FLOAT, v15 FLOAT,
    v16 FLOAT, v17 FLOAT, v18 FLOAT,
    v19 FLOAT, v20 FLOAT, v21 FLOAT,
    v22 FLOAT, v23 FLOAT, v24 FLOAT,
    v25 FLOAT, v26 FLOAT, v27 FLOAT, v28 FLOAT,
    amount DECIMAL(12,2),
    class INT                         -- 0=legit, 1=fraud
);

-- ============================================================
-- ANALYTICAL QUERIES
-- ============================================================

-- 1. Fraud Overview
SELECT
    class,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 4) AS pct,
    ROUND(AVG(amount), 2) AS avg_amount,
    ROUND(MAX(amount), 2) AS max_amount
FROM transactions
GROUP BY class;


-- 2. Fraud by Transaction Amount Bucket
SELECT
    CASE
        WHEN amount < 10   THEN '< $10'
        WHEN amount < 100  THEN '$10–$100'
        WHEN amount < 500  THEN '$100–$500'
        WHEN amount < 1000 THEN '$500–$1000'
        ELSE '> $1000'
    END AS amount_bucket,
    COUNT(*) AS total,
    SUM(class) AS frauds,
    ROUND(SUM(class) * 100.0 / COUNT(*), 3) AS fraud_rate_pct
FROM transactions
GROUP BY amount_bucket
ORDER BY fraud_rate_pct DESC;


-- 3. Time-of-Day Fraud Pattern
-- Assuming time is seconds elapsed (0 = midnight)
SELECT
    FLOOR((transaction_time % 86400) / 3600)::INT AS hour_of_day,
    COUNT(*) AS total,
    SUM(class) AS fraud_count,
    ROUND(SUM(class) * 100.0 / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;


-- 4. High Risk Transactions (for investigation)
SELECT
    transaction_id,
    amount,
    transaction_time,
    class
FROM transactions
WHERE class = 1 OR amount > 1000
ORDER BY amount DESC
LIMIT 100;


-- 5. Rolling Fraud Detection (Window Function)
SELECT
    transaction_id,
    amount,
    class,
    SUM(class) OVER (
        ORDER BY transaction_time
        ROWS BETWEEN 99 PRECEDING AND CURRENT ROW
    ) AS fraud_in_last_100_txns
FROM transactions
ORDER BY transaction_time;


-- 6. Percentile Analysis of Fraud Amounts
SELECT
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) AS p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY amount) AS median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) AS p75,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) AS p95,
    AVG(amount) AS mean_amount
FROM transactions
WHERE class = 1;
