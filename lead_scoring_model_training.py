"""
=============================================================================
LEAD SCORING MODEL - PART 2: FEATURE SELECTION & MODEL TRAINING
=============================================================================
Covers:
- Advanced Feature Selection Methods
- Model Training with Multiple Algorithms
- Hyperparameter Tuning (Bayesian Optimization)
- Cross-Validation & Model Evaluation
- Determining Optimal Feature Set
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import (
    SelectFromModel,
    RFE,
    SequentialFeatureSelector,
    mutual_info_classif,
    f_classif
)
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve,
    auc,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('/home/claude/lead_scoring_outputs', exist_ok=True)

# Set random seed
np.random.seed(42)
pd.set_option('display.max_columns', None)

print("="*80)
print("LEAD SCORING MODEL - FEATURE SELECTION & MODEL TRAINING")
print("="*80)

# Load processed data from Part 1
df = pd.read_csv('/home/claude/lead_scoring_outputs/processed_data.csv')
print(f"\n✓ Loaded processed data: {df.shape}")

# Separate features and target
X = df.drop('converted', axis=1)
y = df['converted']

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")
print(f"✓ Class distribution - Train: {y_train.value_counts(normalize=True).to_dict()}")

# =============================================================================
# SECTION 1: FEATURE SELECTION METHODS
# =============================================================================
print("\n" + "="*80)
print("FEATURE SELECTION - MULTIPLE METHODS")
print("-" * 80)

selected_features_dict = {}

# METHOD 1: STATISTICAL TESTS (F-test)
print("\n1.1: F-TEST (ANOVA) - Statistical Significance")
print("-" * 40)

f_scores, p_values = f_classif(X_train, y_train)
f_stat_df = pd.DataFrame({
    'Feature': X_train.columns,
    'F_Statistic': f_scores,
    'P_Value': p_values
}).sort_values('F_Statistic', ascending=False)

# Select features with p-value < 0.05
f_test_features = f_stat_df[f_stat_df['P_Value'] < 0.05]['Feature'].tolist()
print(f"✓ Features selected (p < 0.05): {len(f_test_features)}")
print(f"  {f_test_features[:10]}...")  # Show first 10
selected_features_dict['F-Test'] = f_test_features

# METHOD 2: MUTUAL INFORMATION
print("\n1.2: MUTUAL INFORMATION - Non-linear Relationships")
print("-" * 40)

mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({
    'Feature': X_train.columns,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

# Select top 80% of features by MI score
mi_threshold = mi_df['MI_Score'].quantile(0.2)  # Top 80%
mi_features = mi_df[mi_df['MI_Score'] >= mi_threshold]['Feature'].tolist()
print(f"✓ Features selected (top 80% by MI): {len(mi_features)}")
print(f"  {mi_features[:10]}...")
selected_features_dict['Mutual Information'] = mi_features

# METHOD 3: RANDOM FOREST FEATURE IMPORTANCE
print("\n1.3: RANDOM FOREST - Model-based Importance")
print("-" * 40)

# Handle class imbalance with class weights
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Cumulative importance threshold: 85%
cumsum_importance = rf_importance['Importance'].cumsum()
cumsum_importance_pct = cumsum_importance / cumsum_importance.iloc[-1] * 100
n_features_85 = (cumsum_importance_pct <= 85).sum() + 1
rf_features = rf_importance.head(n_features_85)['Feature'].tolist()

print(f"✓ Features for 85% cumulative importance: {len(rf_features)}")
print(f"  {rf_features[:10]}...")
selected_features_dict['Random Forest'] = rf_features

# METHOD 4: RECURSIVE FEATURE ELIMINATION (RFE)
print("\n1.4: RECURSIVE FEATURE ELIMINATION (RFE)")
print("-" * 40)

# Use Logistic Regression as estimator
lr_base = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

# RFE with step size
rfe = RFE(
    estimator=lr_base,
    n_features_to_select=25,  # Select top 25 features
    step=3  # Remove 3 features at each step
)
rfe.fit(X_train, y_train)

rfe_features = X_train.columns[rfe.support_].tolist()
print(f"✓ Features selected by RFE: {len(rfe_features)}")
print(f"  {rfe_features}")
selected_features_dict['RFE'] = rfe_features

# METHOD 5: L1 REGULARIZATION (Logistic Regression with L1)
print("\n1.5: L1 REGULARIZATION - Feature Coefficients")
print("-" * 40)

lr_l1 = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=1.0,
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
lr_l1.fit(X_train, y_train)

# Get non-zero coefficients
l1_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_l1.coef_[0],
    'Abs_Coefficient': np.abs(lr_l1.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

l1_features = l1_importance[l1_importance['Abs_Coefficient'] > 0]['Feature'].tolist()
print(f"✓ Features with non-zero coefficients: {len(l1_features)}")
print(f"  {l1_features[:10]}...")
selected_features_dict['L1 Regularization'] = l1_features

# METHOD 6: SEQUENTIAL FORWARD SELECTION (SFS)
print("\n1.6: SEQUENTIAL FORWARD SELECTION (SFS)")
print("-" * 40)
print("This may take 30-60 seconds...")

sfs = SequentialFeatureSelector(
    estimator=RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    n_features_to_select=20,
    direction='forward',
    scoring='roc_auc',
    cv=3
)
sfs.fit(X_train, y_train)

sfs_features = X_train.columns[sfs.support_].tolist()
print(f"✓ Features selected by SFS: {len(sfs_features)}")
print(f"  {sfs_features}")
selected_features_dict['SFS'] = sfs_features

# CONSENSUS: Features selected by MULTIPLE methods
print("\n" + "-" * 40)
print("CONSENSUS FEATURE SELECTION")
print("-" * 40)

# Count how many methods selected each feature
from collections import Counter
all_selected = []
for features in selected_features_dict.values():
    all_selected.extend(features)

feature_consensus = pd.DataFrame({
    'Feature': list(set(all_selected)),
    'Consensus_Count': [all_selected.count(f) for f in list(set(all_selected))]
}).sort_values('Consensus_Count', ascending=False)

# Features selected by 3+ methods (strong consensus)
strong_consensus_features = feature_consensus[feature_consensus['Consensus_Count'] >= 3]['Feature'].tolist()

print(f"\n✓ Features selected by 3+ methods (strong consensus): {len(strong_consensus_features)}")
print(f"  {strong_consensus_features}")

# Summary table
print("\n\nFEATURE SELECTION SUMMARY:")
print("-" * 80)
summary_table = pd.DataFrame({
    'Method': list(selected_features_dict.keys()),
    'N_Features': [len(v) for v in selected_features_dict.values()]
})
print(summary_table.to_string(index=False))

# Visualize consensus
plt.figure(figsize=(12, 8))
top_consensus = feature_consensus.head(20)
plt.barh(top_consensus['Feature'], top_consensus['Consensus_Count'], color='steelblue')
plt.xlabel('Number of Methods (out of 6)')
plt.title('Feature Selection Consensus\n(How many methods selected each feature?)')
plt.axvline(x=3, color='red', linestyle='--', label='Strong Consensus (3+)')
plt.legend()
plt.tight_layout()
plt.savefig('/home/claude/lead_scoring_outputs/03_feature_selection_consensus.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: feature selection consensus visualization")

# =============================================================================
# SECTION 2: COMPARING FEATURE SETS
# =============================================================================
print("\n" + "="*80)
print("COMPARING DIFFERENT FEATURE SETS")
print("-" * 80)

feature_sets = {
    'All Features': X_train.columns.tolist(),
    'Strong Consensus (3+)': strong_consensus_features,
    'Random Forest (85%)': rf_features,
    'RFE (25 features)': rfe_features,
    'SFS (20 features)': sfs_features
}

baseline_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

feature_set_comparison = []

for set_name, features in feature_sets.items():
    if len(features) > 0:
        X_train_subset = X_train[features]
        X_test_subset = X_test[features]
        
        baseline_model.fit(X_train_subset, y_train)
        
        train_auc = roc_auc_score(y_train, baseline_model.predict_proba(X_train_subset)[:, 1])
        test_auc = roc_auc_score(y_test, baseline_model.predict_proba(X_test_subset)[:, 1])
        
        # Cross-validation score
        cv_scores = cross_val_score(
            baseline_model, X_train_subset, y_train,
            cv=5, scoring='roc_auc'
        )
        
        feature_set_comparison.append({
            'Feature_Set': set_name,
            'N_Features': len(features),
            'Train_AUC': train_auc,
            'Test_AUC': test_auc,
            'CV_Mean_AUC': cv_scores.mean(),
            'CV_Std_AUC': cv_scores.std(),
            'Overfitting_Gap': train_auc - test_auc
        })

comparison_df = pd.DataFrame(feature_set_comparison).sort_values('Test_AUC', ascending=False)
print("\nFEATURE SET COMPARISON (with Random Forest baseline):")
print(comparison_df.to_string(index=False))

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AUC comparison
comparison_df.plot(x='Feature_Set', y=['Train_AUC', 'Test_AUC'], kind='bar', ax=axes[0])
axes[0].set_title('Train vs Test AUC by Feature Set')
axes[0].set_ylabel('AUC Score')
axes[0].set_xlabel('')
axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random Baseline')
axes[0].legend()
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# Feature count vs performance
axes[1].scatter(comparison_df['N_Features'], comparison_df['Test_AUC'], s=200, alpha=0.6)
for idx, row in comparison_df.iterrows():
    axes[1].annotate(row['Feature_Set'], 
                    (row['N_Features'], row['Test_AUC']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[1].set_xlabel('Number of Features')
axes[1].set_ylabel('Test AUC')
axes[1].set_title('Feature Count vs Performance')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/lead_scoring_outputs/04_feature_set_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: feature set comparison visualization")

# Select optimal feature set (balance performance and interpretability)
optimal_feature_set = 'Strong Consensus (3+)'  # Good balance
optimal_features = feature_sets[optimal_feature_set]

print(f"\n{'='*80}")
print(f"SELECTED OPTIMAL FEATURE SET: {optimal_feature_set}")
print(f"Number of features: {len(optimal_features)}")
print(f"Expected Test AUC: {comparison_df[comparison_df['Feature_Set']==optimal_feature_set]['Test_AUC'].values[0]:.4f}")
print(f"{'='*80}")

# =============================================================================
# SECTION 3: MODEL TRAINING WITH OPTIMAL FEATURES
# =============================================================================
print("\n" + "="*80)
print("MODEL TRAINING WITH OPTIMAL FEATURES")
print("-" * 80)

# Prepare data with optimal features
X_train_opt = X_train[optimal_features]
X_test_opt = X_test[optimal_features]

print(f"\n✓ Using {len(optimal_features)} optimal features:")
for i, f in enumerate(optimal_features, 1):
    print(f"  {i}. {f}")

# =============================================================================
# MODEL 1: LOGISTIC REGRESSION
# =============================================================================
print("\n\n[MODEL 1] LOGISTIC REGRESSION")
print("-" * 80)

# Hyperparameter tuning for Logistic Regression
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'max_iter': [1000, 2000]
}

# Use class weights to handle imbalance
lr_base = LogisticRegression(random_state=42, class_weight='balanced')

grid_search_lr = GridSearchCV(
    lr_base,
    param_grid_lr,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

grid_search_lr.fit(X_train_opt, y_train)

print(f"✓ Best hyperparameters: {grid_search_lr.best_params_}")
print(f"✓ Best CV AUC: {grid_search_lr.best_score_:.4f}")

lr_model = grid_search_lr.best_estimator_
y_pred_lr = lr_model.predict(X_test_opt)
y_pred_proba_lr = lr_model.predict_proba(X_test_opt)[:, 1]

lr_auc = roc_auc_score(y_test, y_pred_proba_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)

print(f"\nTest Set Performance:")
print(f"  AUC: {lr_auc:.4f}")
print(f"  F1-Score: {lr_f1:.4f}")
print(f"  Precision: {lr_precision:.4f}")
print(f"  Recall: {lr_recall:.4f}")

# Get feature coefficients (these are your WEIGHTS!)
lr_coefficients = pd.DataFrame({
    'Feature': optimal_features,
    'Coefficient': lr_model.coef_[0],
    'Abs_Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\nFeature Weights (Coefficients):")
print(lr_coefficients.to_string(index=False))

# =============================================================================
# MODEL 2: GRADIENT BOOSTING
# =============================================================================
print("\n\n[MODEL 2] GRADIENT BOOSTING")
print("-" * 80)

# Hyperparameter tuning
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'min_samples_split': [5, 10]
}

gb_base = GradientBoostingClassifier(random_state=42)

# Use RandomizedSearchCV instead of GridSearchCV for faster results
random_search_gb = RandomizedSearchCV(
    gb_base,
    param_grid_gb,
    n_iter=20,  # Test 20 random combinations
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

random_search_gb.fit(X_train_opt, y_train)

print(f"✓ Best hyperparameters: {random_search_gb.best_params_}")
print(f"✓ Best CV AUC: {random_search_gb.best_score_:.4f}")

gb_model = random_search_gb.best_estimator_
y_pred_gb = gb_model.predict(X_test_opt)
y_pred_proba_gb = gb_model.predict_proba(X_test_opt)[:, 1]

gb_auc = roc_auc_score(y_test, y_pred_proba_gb)
gb_f1 = f1_score(y_test, y_pred_gb)
gb_precision = precision_score(y_test, y_pred_gb)
gb_recall = recall_score(y_test, y_pred_gb)

print(f"\nTest Set Performance:")
print(f"  AUC: {gb_auc:.4f}")
print(f"  F1-Score: {gb_f1:.4f}")
print(f"  Precision: {gb_precision:.4f}")
print(f"  Recall: {gb_recall:.4f}")

gb_importance = pd.DataFrame({
    'Feature': optimal_features,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nFeature Importance:")
print(gb_importance.to_string(index=False))

# =============================================================================
# MODEL 3: XGBOOST
# =============================================================================
print("\n\n[MODEL 3] XGBOOST")
print("-" * 80)

# Calculate scale_pos_weight to handle class imbalance
n_neg, n_pos = np.bincount(y_train)
scale_pos_weight = n_neg / n_pos

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1]
}

xgb_base = xgb.XGBClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    tree_method='hist',
    device='cpu'
)

random_search_xgb = RandomizedSearchCV(
    xgb_base,
    param_grid_xgb,
    n_iter=20,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

random_search_xgb.fit(X_train_opt, y_train)

print(f"✓ Best hyperparameters: {random_search_xgb.best_params_}")
print(f"✓ Best CV AUC: {random_search_xgb.best_score_:.4f}")

xgb_model = random_search_xgb.best_estimator_
y_pred_xgb = xgb_model.predict(X_test_opt)
y_pred_proba_xgb = xgb_model.predict_proba(X_test_opt)[:, 1]

xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)

print(f"\nTest Set Performance:")
print(f"  AUC: {xgb_auc:.4f}")
print(f"  F1-Score: {xgb_f1:.4f}")
print(f"  Precision: {xgb_precision:.4f}")
print(f"  Recall: {xgb_recall:.4f}")

xgb_importance = pd.DataFrame({
    'Feature': optimal_features,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nFeature Importance:")
print(xgb_importance.to_string(index=False))

# =============================================================================
# SECTION 4: MODEL COMPARISON & SELECTION
# =============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON & FINAL SELECTION")
print("-" * 80)

model_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Gradient Boosting', 'XGBoost'],
    'AUC': [lr_auc, gb_auc, xgb_auc],
    'F1-Score': [lr_f1, gb_f1, xgb_f1],
    'Precision': [lr_precision, gb_precision, xgb_precision],
    'Recall': [lr_recall, gb_recall, xgb_recall]
}).sort_values('AUC', ascending=False)

print("\nPERFORMANCE COMPARISON:")
print(model_comparison.to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# AUC Scores
model_comparison.plot(x='Model', y='AUC', kind='bar', ax=axes[0, 0], legend=False)
axes[0, 0].set_title('AUC Scores')
axes[0, 0].set_ylabel('AUC')
axes[0, 0].set_ylim([0.5, 1.0])
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')

# F1-Score
model_comparison.plot(x='Model', y='F1-Score', kind='bar', ax=axes[0, 1], legend=False)
axes[0, 1].set_title('F1-Score')
axes[0, 1].set_ylabel('F1-Score')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')

# Precision vs Recall
axes[1, 0].scatter(model_comparison['Recall'], model_comparison['Precision'], s=200, alpha=0.6)
for idx, row in model_comparison.iterrows():
    axes[1, 0].annotate(row['Model'], 
                       (row['Recall'], row['Precision']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision-Recall Trade-off')
axes[1, 0].grid(True, alpha=0.3)

# All metrics radar (approximate)
models = model_comparison['Model'].tolist()
metrics = ['AUC', 'F1-Score', 'Precision', 'Recall']
normalized_metrics = model_comparison[metrics].div(model_comparison[metrics].max(axis=0), axis=1)

x = np.arange(len(metrics))
width = 0.25

for i, model in enumerate(models):
    axes[1, 1].bar(x + i*width, normalized_metrics.iloc[i], width, label=model)

axes[1, 1].set_xlabel('Metrics')
axes[1, 1].set_ylabel('Normalized Score (0-1)')
axes[1, 1].set_title('All Metrics Comparison (Normalized)')
axes[1, 1].set_xticks(x + width)
axes[1, 1].set_xticklabels(metrics, rotation=45, ha='right')
axes[1, 1].legend()
axes[1, 1].set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('/home/claude/lead_scoring_outputs/05_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model comparison visualization")

# Select best model
best_model_name = model_comparison.iloc[0]['Model']
if best_model_name == 'Logistic Regression':
    best_model = lr_model
    best_feature_importance = lr_coefficients
elif best_model_name == 'Gradient Boosting':
    best_model = gb_model
    best_feature_importance = gb_importance
else:
    best_model = xgb_model
    best_feature_importance = xgb_importance

print(f"\n{'='*80}")
print(f"BEST MODEL SELECTED: {best_model_name}")
print(f"AUC Score: {model_comparison.iloc[0]['AUC']:.4f}")
print(f"{'='*80}")

# Save models
import pickle

with open('/home/claude/lead_scoring_outputs/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('/home/claude/lead_scoring_outputs/lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('/home/claude/lead_scoring_outputs/gb_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)

with open('/home/claude/lead_scoring_outputs/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

print("✓ Models saved as pickle files")

# Save optimal features
pd.DataFrame({'Feature': optimal_features}).to_csv(
    '/home/claude/lead_scoring_outputs/optimal_features.csv', index=False
)
print("✓ Optimal features saved")

# Save feature importance
best_feature_importance.to_csv(
    '/home/claude/lead_scoring_outputs/feature_importance.csv', index=False
)
print("✓ Feature importance saved")

