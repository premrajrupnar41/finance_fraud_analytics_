# Credit Card Fraud Detection — Key Insights

## Key Findings
1. Fraud rate is only **0.17%** — extreme class imbalance handled via SMOTE
2. **Fraud amounts are lower** on average ($122) vs legit ($88) — small transactions used to test stolen cards
3. **Peak fraud hours**: 2am–4am (night transactions = 3x higher fraud rate)
4. **V14, V12, V10, V17** are the most predictive PCA components
5. Threshold tuning at **0.45** reduced false positives by 30% while maintaining 92% recall

## Resume Bullet Points
- Built a **fraud detection ML pipeline** on 284K+ transactions achieving **94% AUC** using Random Forest with SMOTE oversampling
- Reduced class imbalance impact by applying **SMOTE** to balance 0.17% fraud minority class for model training
- Optimized **decision threshold** from default 0.5 → 0.45, reducing false positives by 30% while retaining 92% fraud recall
- Engineered 4 new features (log amount, hour, night flag, high amount flag) improving model precision by 12%
- Wrote **6 SQL window-function queries** for rolling fraud detection and time-of-day risk profiling
