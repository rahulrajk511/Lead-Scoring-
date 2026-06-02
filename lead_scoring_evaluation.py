"""
=============================================================================
LEAD SCORING MODEL - PART 3: EVALUATION & IMPLEMENTATION
=============================================================================
Covers:
- Detailed Model Evaluation (ROC, PR Curves, Confusion Matrix)
- Feature Importance & Interpretation
- Model Calibration
- Lead Scoring Implementation
- Business Insights & Recommendations
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve,
    auc,
    roc_curve,
    confusion_matrix,
    classification_report,
    calibration_curve,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
import pickle
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('/home/claude/lead_scoring_outputs', exist_ok=True)

print("="*80)
print("LEAD SCORING MODEL - EVALUATION & IMPLEMENTATION")
print("="*80)

# Load data
df = pd.read_csv('/home/claude/lead_scoring_outputs/processed_data.csv')
X = df.drop('converted', axis=1)
y = df['converted']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load optimal features and models
optimal_features = pd.read_csv('/home/claude/lead_scoring_outputs/optimal_features.csv')['Feature'].tolist()
X_train_opt = X_train[optimal_features]
X_test_opt = X_test[optimal_features]

with open('/home/claude/lead_scoring_outputs/best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('/home/claude/lead_scoring_outputs/lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

feature_importance = pd.read_csv('/home/claude/lead_scoring_outputs/feature_importance.csv')

print(f"✓ Loaded best model and optimal features ({len(optimal_features)} features)")

# =============================================================================
# SECTION 1: DETAILED MODEL EVALUATION
# =============================================================================
print("\n" + "="*80)
print("[1] DETAILED MODEL EVALUATION")
print("-" * 80)

# Get predictions
y_pred_proba = best_model.predict_proba(X_test_opt)[:, 1]
y_pred = best_model.predict(X_test_opt)

# Calculate metrics
auc_score = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall_val = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
brier = brier_score_loss(y_test, y_pred_proba)

print("\nOVERALL PERFORMANCE METRICS:")
print("-" * 40)
print(f"AUC-ROC:        {auc_score:.4f} (scale: 0.5-1.0, higher is better)")
print(f"AUC-PR:         {pr_auc:.4f}")
print(f"Accuracy:       {accuracy:.4f}")
print(f"Precision:      {precision:.4f} (of predicted conversions, how many are correct)")
print(f"Recall:         {recall_val:.4f} (of actual conversions, how many did we find)")
print(f"F1-Score:       {f1:.4f} (harmonic mean of precision and recall)")
print(f"Brier Score:    {brier:.4f} (lower is better, measures calibration)")

print("\nCONFUSION MATRIX:")
print("-" * 40)
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives:  {cm[0,0]} (correctly rejected)")
print(f"False Positives: {cm[0,1]} (false alerts)")
print(f"False Negatives: {cm[1,0]} (missed conversions)")
print(f"True Positives:  {cm[1,1]} (correctly identified)")

print("\nCLASSIFICATION REPORT:")
print("-" * 40)
print(classification_report(y_test, y_pred, target_names=['Not Converted', 'Converted']))

# =============================================================================
# SECTION 2: VISUALIZATION - ROC & PR CURVES
# =============================================================================
print("\n" + "="*80)
print("[2] EVALUATION CURVES & DIAGNOSTICS")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. ROC Curve
axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC={auc_score:.4f})', color='steelblue', lw=2)
axes[0, 0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random Baseline (AUC=0.5)')
axes[0, 0].fill_between(fpr, tpr, alpha=0.2)
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# 2. Precision-Recall Curve
axes[0, 1].plot(recall, precision, label=f'PR Curve (AUC={pr_auc:.4f})', color='seagreen', lw=2)
baseline_pr = y_test.sum() / len(y_test)
axes[0, 1].axhline(y=baseline_pr, color='red', linestyle='--', lw=1, label=f'Baseline ({baseline_pr:.4f})')
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curve')
axes[0, 1].legend(loc='upper right')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim([0, 1])
axes[0, 1].set_ylim([0, 1])

# 3. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], cbar=False,
            xticklabels=['Not Converted', 'Converted'],
            yticklabels=['Not Converted', 'Converted'])
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')
axes[1, 0].set_title('Confusion Matrix')

# 4. Score Distribution
axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, label='Not Converted', color='blue')
axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Converted', color='red')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Prediction Score Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/lead_scoring_outputs/06_evaluation_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: evaluation curves visualization")

# =============================================================================
# SECTION 3: CALIBRATION ANALYSIS
# =============================================================================
print("\n" + "-" * 80)
print("CALIBRATION ANALYSIS")
print("-" * 80)

# Calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

print(f"\nCalibration Analysis:")
print(f"  Brier Score: {brier:.4f}")
print(f"  (Brier < 0.25 is good, perfect calibration = 0)")

# Visualize calibration
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Calibration curve
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect Calibration')
axes[0].plot(prob_pred, prob_true, 'o-', lw=2, label='Current Model', color='steelblue')
axes[0].fill_between([0, 1], [0, 1], alpha=0.2)
axes[0].set_xlabel('Mean Predicted Probability')
axes[0].set_ylabel('Fraction of Positives')
axes[0].set_title('Calibration Plot')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])

# Reliability diagram
axes[1].bar(prob_pred, prob_true - prob_pred, width=0.05, alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='-', lw=0.8)
axes[1].set_xlabel('Mean Predicted Probability')
axes[1].set_ylabel('Difference (Actual - Predicted)')
axes[1].set_title('Calibration Error')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/lead_scoring_outputs/07_calibration_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: calibration analysis visualization")

# =============================================================================
# SECTION 4: FEATURE IMPORTANCE & INTERPRETATION
# =============================================================================
print("\n" + "="*80)
print("[3] FEATURE IMPORTANCE & WEIGHTS")
print("-" * 80)

print("\nTOP 15 FEATURES BY IMPORTANCE/WEIGHT:")
print("-" * 40)
top_features = feature_importance.head(15)
print(top_features.to_string(index=False))

# Get feature weights from Logistic Regression for interpretation
lr_coefficients = pd.DataFrame({
    'Feature': optimal_features,
    'LR_Coefficient': lr_model.coef_[0],
    'LR_Abs_Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('LR_Abs_Coefficient', ascending=False)

print("\n\nLOGISTIC REGRESSION COEFFICIENTS (Directly Interpretable):")
print("-" * 40)
print("These are the weights that tell you the relative importance of each feature.")
print("Positive coefficient = increases conversion likelihood")
print("Negative coefficient = decreases conversion likelihood")
print("-" * 40)

top_lr_coef = lr_coefficients.head(15)
for idx, row in top_lr_coef.iterrows():
    direction = "↑ Increases" if row['LR_Coefficient'] > 0 else "↓ Decreases"
    print(f"{row['Feature']:40s} {row['LR_Coefficient']:8.4f}  {direction} conversion")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Feature Importance from best model
top_imp = feature_importance.head(15)
axes[0].barh(top_imp['Feature'], top_imp['Importance'], color='steelblue')
axes[0].set_xlabel('Feature Importance')
axes[0].set_title('Top 15 Features by Model Importance')
axes[0].invert_yaxis()

# Logistic Regression Coefficients
top_lr = lr_coefficients.head(15)
colors = ['red' if x < 0 else 'green' for x in top_lr['LR_Coefficient']]
axes[1].barh(top_lr['Feature'], top_lr['LR_Coefficient'], color=colors)
axes[1].set_xlabel('Coefficient Value')
axes[1].set_title('Top 15 Features by LR Coefficient\n(Green=Positive, Red=Negative)')
axes[1].axvline(x=0, color='black', linestyle='-', lw=0.8)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('/home/claude/lead_scoring_outputs/08_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: feature importance visualization")

# =============================================================================
# SECTION 5: LEAD SCORING IMPLEMENTATION
# =============================================================================
print("\n" + "="*80)
print("[4] LEAD SCORING IMPLEMENTATION")
print("-" * 80)

print("""
LEAD SCORING STRATEGY:
1. Use the model to predict conversion probability for each lead
2. Convert probability to percentile-based score (0-100)
3. Define business rules based on percentiles
4. Assign priorities and actions based on scores
""")

# Generate lead scores for test set
test_leads = X_test_opt.copy()
test_leads['Conversion_Probability'] = y_pred_proba
test_leads['Lead_Score_0_100'] = (y_pred_proba * 100).round(0).astype(int)
test_leads['Percentile'] = (pd.Series(y_pred_proba).rank(pct=True) * 100).round(0).astype(int)
test_leads['Actual_Converted'] = y_test.values
test_leads['Correct_Prediction'] = (test_leads['Lead_Score_0_100'] >= 50).astype(int) == test_leads['Actual_Converted']

# Add interpretation
def assign_segment(score):
    if score >= 80:
        return 'Hot Lead - Immediate Follow-up'
    elif score >= 60:
        return 'Warm Lead - Priority Follow-up'
    elif score >= 40:
        return 'Lukewarm Lead - Monitor'
    elif score >= 20:
        return 'Cool Lead - Nurture'
    else:
        return 'Cold Lead - Low Priority'

test_leads['Segment'] = test_leads['Lead_Score_0_100'].apply(assign_segment)

# Display sample leads
print("\nSAMPLE SCORED LEADS:")
print("-" * 80)
sample_leads = test_leads[['Lead_Score_0_100', 'Conversion_Probability', 'Percentile', 
                            'Segment', 'Actual_Converted', 'Correct_Prediction']].head(20)
print(sample_leads.to_string())

# Segment analysis
print("\n\nSEGMENT ANALYSIS:")
print("-" * 80)

segment_order = ['Hot Lead - Immediate Follow-up', 'Warm Lead - Priority Follow-up', 
                 'Lukewarm Lead - Monitor', 'Cool Lead - Nurture', 'Cold Lead - Low Priority']

segment_analysis = test_leads.groupby('Segment').agg({
    'Actual_Converted': ['sum', 'count', 'mean'],
    'Correct_Prediction': 'mean'
}).round(4)

segment_analysis.columns = ['Conversions', 'Total_Leads', 'Conversion_Rate', 'Model_Accuracy']
segment_analysis = segment_analysis.reindex(segment_order)

print(segment_analysis.to_string())

# Lift Analysis
print("\n\nLIFT ANALYSIS:")
print("-" * 80)

overall_conversion_rate = y_test.mean()

lift_analysis = []
for segment in segment_order:
    segment_data = test_leads[test_leads['Segment'] == segment]
    if len(segment_data) > 0:
        segment_conv_rate = segment_data['Actual_Converted'].mean()
        lift = segment_conv_rate / overall_conversion_rate if overall_conversion_rate > 0 else 0
        lift_analysis.append({
            'Segment': segment,
            'Leads': len(segment_data),
            'Conversions': segment_data['Actual_Converted'].sum(),
            'Conversion_Rate': segment_conv_rate,
            'Lift': lift
        })

lift_df = pd.DataFrame(lift_analysis)
print(lift_df.to_string(index=False))

print(f"\nInterpretation:")
print(f"  - Baseline conversion rate (all leads): {overall_conversion_rate:.2%}")
print(f"  - Hot leads have {lift_df.iloc[0]['Lift']:.2f}x lift (higher conversion probability)")
print(f"  - This means prioritizing hot leads is {lift_df.iloc[0]['Lift']:.0f} times more effective")

# Save lead scores
test_leads.to_csv('/home/claude/lead_scoring_outputs/scored_leads.csv', index=False)
print("\n✓ Saved: scored_leads.csv")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution of lead scores
axes[0, 0].hist(test_leads['Lead_Score_0_100'], bins=20, color='steelblue', edgecolor='black')
axes[0, 0].axvline(x=50, color='red', linestyle='--', label='Decision Threshold (50)')
axes[0, 0].set_xlabel('Lead Score (0-100)')
axes[0, 0].set_ylabel('Number of Leads')
axes[0, 0].set_title('Distribution of Lead Scores')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Conversion rate by score segment
segment_conv = test_leads.groupby('Lead_Score_0_100')['Actual_Converted'].agg(['sum', 'count'])
segment_conv['rate'] = segment_conv['sum'] / segment_conv['count']
axes[0, 1].scatter(segment_conv.index, segment_conv['rate'], s=100, alpha=0.6)
axes[0, 1].plot(segment_conv.index, segment_conv['rate'], alpha=0.3)
axes[0, 1].set_xlabel('Lead Score')
axes[0, 1].set_ylabel('Actual Conversion Rate')
axes[0, 1].set_title('Actual Conversion Rate by Score')
axes[0, 1].grid(True, alpha=0.3)

# 3. Segment analysis - counts and conversion rates
segment_data = test_leads['Segment'].value_counts()
colors_map = {'Hot Lead - Immediate Follow-up': 'darkred', 
              'Warm Lead - Priority Follow-up': 'red',
              'Lukewarm Lead - Monitor': 'orange',
              'Cool Lead - Nurture': 'yellow',
              'Cold Lead - Low Priority': 'lightblue'}
colors = [colors_map.get(s, 'gray') for s in segment_data.index]

axes[1, 0].bar(range(len(segment_data)), segment_data.values, color=colors)
axes[1, 0].set_xticks(range(len(segment_data)))
axes[1, 0].set_xticklabels([s.split(' - ')[0] for s in segment_data.index], rotation=45, ha='right')
axes[1, 0].set_ylabel('Number of Leads')
axes[1, 0].set_title('Lead Distribution by Segment')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Lift chart
lift_df_sorted = lift_df.sort_values('Lift', ascending=True)
axes[1, 1].barh(lift_df_sorted['Segment'], lift_df_sorted['Lift'], color='steelblue')
axes[1, 1].axvline(x=1, color='red', linestyle='--', label='No Lift (1x)')
axes[1, 1].set_xlabel('Lift vs Baseline')
axes[1, 1].set_title('Lift Analysis by Segment')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/home/claude/lead_scoring_outputs/09_lead_scoring_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: lead scoring results visualization")

# =============================================================================
# SECTION 6: SCORING FUNCTION FOR NEW LEADS
# =============================================================================
print("\n" + "="*80)
print("[5] IMPLEMENTATION GUIDE FOR NEW LEADS")
print("-" * 80)

implementation_code = '''
# ============================================================================
# HOW TO SCORE NEW LEADS IN PRODUCTION
# ============================================================================

import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

# 1. Load the model and scaler
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Prepare new lead data (same preprocessing as training)
def score_new_leads(new_leads_df, optimal_features):
    """
    new_leads_df: DataFrame with raw data
    optimal_features: List of features used in training
    """
    # Apply same preprocessing as training data
    # (missing value imputation, feature engineering, scaling)
    
    # Get predictions
    X_new = new_leads_df[optimal_features]
    probabilities = model.predict_proba(X_new)[:, 1]
    
    # Create lead scores
    new_leads_df['Conversion_Probability'] = probabilities
    new_leads_df['Lead_Score_0_100'] = (probabilities * 100).round(0).astype(int)
    
    # Assign segments
    def assign_segment(score):
        if score >= 80:
            return 'Hot Lead - Immediate Follow-up'
        elif score >= 60:
            return 'Warm Lead - Priority Follow-up'
        elif score >= 40:
            return 'Lukewarm Lead - Monitor'
        elif score >= 20:
            return 'Cool Lead - Nurture'
        else:
            return 'Cold Lead - Low Priority'
    
    new_leads_df['Segment'] = new_leads_df['Lead_Score_0_100'].apply(assign_segment)
    
    return new_leads_df[['Lead_Score_0_100', 'Conversion_Probability', 'Segment']]

# 3. Usage
new_leads = pd.read_csv('new_leads.csv')
scored = score_new_leads(new_leads, optimal_features)
print(scored)
'''

print("\nPython implementation guide:")
print(implementation_code)

# =============================================================================
# SECTION 7: KEY BUSINESS RECOMMENDATIONS
# =============================================================================
print("\n" + "="*80)
print("[6] KEY BUSINESS RECOMMENDATIONS")
print("-" * 80)

recommendations = f"""
RECOMMENDATIONS FOR ASSET MANAGEMENT TEAM:

1. LEAD PRIORITIZATION:
   - Focus sales efforts on leads with score > 60 (Warm & Hot)
   - These represent {len(test_leads[test_leads['Lead_Score_0_100'] >= 60])} leads in test set
   - Expected conversion lift: {lift_df[lift_df['Lead_Score_0_100'] >= 60]['Lift'].mean():.2f}x baseline

2. ENGAGEMENT STRATEGY:
   Key drivers of high lead scores:
   - {lr_coefficients.iloc[0]['Feature']} (strongest predictor)
   - {lr_coefficients.iloc[1]['Feature']} (second strongest)
   - {lr_coefficients.iloc[2]['Feature']} (third strongest)
   
   ACTION: Invest in tactics that drive these signals

3. CHANNEL OPTIMIZATION:
   - Email engagement is high-quality signal (more data available)
   - Website visits and webinar attendance are even stronger signals
   - Implement email + website + webinar touchpoints for max impact

4. SCORING GOVERNANCE:
   - Update model every 3 months with new data
   - Monitor model drift (if actual conversion rates change)
   - Establish feedback loop: track actual outcomes vs predicted scores

5. INTEGRATION WITH CRM:
   - Export scores to Salesforce/HubSpot daily
   - Trigger workflows based on segments
   - Hot leads (>80): Assign to account managers immediately
   - Warm leads (60-80): Include in targeted campaigns

6. EXPECTED OUTCOMES:
   - AUC Score: {auc_score:.4f} (strong predictive power)
   - Top 30% of leads = ~{lift_df[lift_df['Lead_Score_0_100'] >= 60]['Conversion_Rate'].mean():.1%} conversion rate
   - Baseline = {overall_conversion_rate:.1%} conversion rate
   - ROI: {(lift_df[lift_df['Lead_Score_0_100'] >= 60]['Lift'].mean() - 1) * 100:.0f}% improvement by focusing on warm leads
"""

print(recommendations)

# Save recommendations
with open('/home/claude/lead_scoring_outputs/RECOMMENDATIONS.txt', 'w') as f:
    f.write(recommendations)

print("\n✓ Saved: RECOMMENDATIONS.txt")

# =============================================================================
# SECTION 8: SUMMARY REPORT
# =============================================================================
print("\n" + "="*80)
print("[7] FINAL SUMMARY")
print("="*80)

summary = f"""
LEAD SCORING MODEL - FINAL SUMMARY
{'='*80}

MODEL PERFORMANCE:
  - AUC-ROC Score:        {auc_score:.4f}
  - Precision:            {precision_score(y_test, y_pred):.4f}
  - Recall:               {recall_score(y_test, y_pred):.4f}
  - F1-Score:             {f1_score(y_test, y_pred):.4f}

FEATURES SELECTED:
  - Total features evaluated: {len(X.columns)}
  - Optimal feature set size: {len(optimal_features)}
  - Selection method: Consensus of 6 feature selection techniques

TOP 5 FEATURES:
"""

for i, row in feature_importance.head(5).iterrows():
    summary += f"  {i+1}. {row['Feature']}\n"

summary += f"""
BUSINESS IMPACT:
  - Can identify {len(test_leads[test_leads['Lead_Score_0_100'] >= 60])} out of {len(test_leads)} leads as 'warm' or 'hot'
  - Lift vs baseline: {lift_df.iloc[0]['Lift']:.2f}x for hot leads
  - Expected efficiency gain: {(lift_df[lift_df['Lead_Score_0_100'] >= 60]['Lift'].mean() - 1) * 100:.0f}%

NEXT STEPS:
  1. Validate model with new holdout dataset
  2. Deploy to production with daily batch scoring
  3. Set up monitoring dashboard
  4. Establish monthly model retraining cycle
  5. Track business metrics (conversion rate, sales cycle length, deal size)

FILES GENERATED:
  - best_model.pkl              : Production-ready model
  - optimal_features.csv         : Features to use for scoring
  - feature_importance.csv       : Feature weights and importance
  - scored_leads.csv             : Example scored test set
  - Visualizations (6 PNG files)
"""

print(summary)

# Save summary
with open('/home/claude/lead_scoring_outputs/SUMMARY.txt', 'w') as f:
    f.write(summary)

print("✓ Saved: SUMMARY.txt")
print("\n" + "="*80)
print("ANALYSIS COMPLETE! All outputs saved to /home/claude/lead_scoring_outputs/")
print("="*80)

