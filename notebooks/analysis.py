# ============================================================
# CREDIT CARD FRAUD DETECTION — Full ML Pipeline
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
df = pd.read_csv('../data/creditcard.csv')
print(f"Shape: {df.shape}")
print(f"\nFraud Distribution:\n{df['Class'].value_counts()}")
print(f"Fraud Rate: {df['Class'].mean()*100:.3f}%")

# ============================================================
# STEP 2: EDA
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Fraud Detection EDA', fontsize=16, fontweight='bold')

# Fraud distribution
axes[0,0].bar(['Legit', 'Fraud'], df['Class'].value_counts().values,
              color=['#2196F3', '#F44336'])
axes[0,0].set_title('Class Distribution (Highly Imbalanced)')
axes[0,0].set_yscale('log')

# Amount distribution: fraud vs legit
df[df['Class']==0]['Amount'].clip(0,500).plot(
    kind='hist', bins=50, alpha=0.6, label='Legit', ax=axes[0,1], color='#2196F3')
df[df['Class']==1]['Amount'].clip(0,500).plot(
    kind='hist', bins=50, alpha=0.6, label='Fraud', ax=axes[0,1], color='#F44336')
axes[0,1].set_title('Transaction Amount: Legit vs Fraud')
axes[0,1].set_xlabel('Amount ($)')
axes[0,1].legend()

# Time pattern
df['Hour'] = (df['Time'] % 86400) // 3600
fraud_by_hour = df.groupby('Hour')['Class'].mean() * 100
fraud_by_hour.plot(kind='bar', ax=axes[1,0], color='#FF9800')
axes[1,0].set_title('Fraud Rate by Hour of Day')
axes[1,0].set_ylabel('Fraud Rate (%)')

# Correlation heatmap (top features)
top_feats = ['V1','V2','V3','V4','V10','V12','V14','V17','Amount','Class']
corr = df[top_feats].corr()
sns.heatmap(corr, ax=axes[1,1], cmap='RdBu_r', center=0, fmt='.2f',
            annot=True, linewidths=0.5)
axes[1,1].set_title('Feature Correlation (Top Features)')

plt.tight_layout()
plt.savefig('../reports/eda_charts.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
df['Amount_Log'] = np.log1p(df['Amount'])
df['Hour'] = (df['Time'] % 86400) // 3600
df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 5)).astype(int)
df['High_Amount'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)

features = [c for c in df.columns if c.startswith('V')] + \
           ['Amount_Log', 'Hour', 'Is_Night', 'High_Amount']

X = df[features]
y = df['Class']

# ============================================================
# STEP 4: HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train)
print(f"\nAfter SMOTE: {y_train_bal.value_counts().to_dict()}")

# ============================================================
# STEP 5: MODEL TRAINING & EVALUATION
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

results = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Evaluation', fontsize=14, fontweight='bold')

colors = ['#2196F3', '#F44336']
for i, (name, model) in enumerate(models.items()):
    model.fit(X_train_bal, y_train_bal)
    proba = model.predict_proba(X_test_sc)[:, 1]
    pred = model.predict(X_test_sc)
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    results[name] = {'auc': auc, 'ap': ap}

    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', lw=2, color=colors[i])
    print(f"\n{name}  AUC: {auc:.4f}  AP: {ap:.4f}")
    print(classification_report(y_test, pred))

axes[0].plot([0,1],[0,1],'k--', lw=1)
axes[0].set_title('ROC Curves')
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Precision-Recall (more meaningful for imbalanced data)
for i, (name, model) in enumerate(models.items()):
    proba = model.predict_proba(X_test_sc)[:,1]
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    axes[1].plot(recall, precision, label=f'{name} (AP={ap:.3f})', lw=2, color=colors[i])
axes[1].set_title('Precision-Recall Curve')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Feature Importance (RF)
rf_model = models['Random Forest']
fi = pd.Series(rf_model.feature_importances_, index=features).nlargest(10)
fi.sort_values().plot(kind='barh', ax=axes[2], color='#FF9800')
axes[2].set_title('Top 10 Feature Importances (RF)')

plt.tight_layout()
plt.savefig('../reports/model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# STEP 6: THRESHOLD OPTIMIZATION
# ============================================================
best_model = models['Random Forest']
proba_test = best_model.predict_proba(X_test_sc)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.05)
precisions, recalls, f1s = [], [], []
for t in thresholds:
    pred = (proba_test >= t).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score
    precisions.append(precision_score(y_test, pred, zero_division=0))
    recalls.append(recall_score(y_test, pred, zero_division=0))
    f1s.append(f1_score(y_test, pred, zero_division=0))

best_t = thresholds[np.argmax(f1s)]
print(f"\nOptimal Threshold: {best_t:.2f}  (Best F1: {max(f1s):.4f})")

# Export predictions
X_test_df = X_test.copy()
X_test_df['fraud_probability'] = proba_test
X_test_df['predicted_fraud'] = (proba_test >= best_t).astype(int)
X_test_df['actual_class'] = y_test.values
X_test_df.to_csv('../reports/fraud_predictions.csv', index=False)

print("\n✅ PIPELINE COMPLETE")
print(f"   Best AUC: {roc_auc_score(y_test, proba_test):.4f}")
print(f"   Optimal Threshold: {best_t:.2f}")
