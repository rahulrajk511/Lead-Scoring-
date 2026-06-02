"""
=============================================================================
LEAD SCORING MODEL - COMPLETE EXECUTION GUIDE
=============================================================================
This guide shows you how to:
1. Run the complete pipeline
2. Adapt to your actual data
3. Interpret results
4. Deploy to production
=============================================================================
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║         LEAD SCORING MODEL - EXECUTION GUIDE FOR ASSET MANAGEMENT          ║
║                    Complete End-to-End Implementation                       ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# SECTION 1: STEP-BY-STEP EXECUTION INSTRUCTIONS
# =============================================================================
print("""
{'='*80}
SECTION 1: RUNNING THE COMPLETE PIPELINE
{'='*80}

STEP 1: Execute Data Preprocessing (Part 1)
-------------------------------------------
This script handles:
✓ Data quality assessment
✓ Missing value treatment (KNN imputation for 30% missing website data)
✓ Feature engineering (engagement scores, recency scores)
✓ Outlier detection and capping
✓ Feature normalization with RobustScaler

Command:
  python lead_scoring_complete_guide.py

Expected Output:
  - 01_missing_data_analysis.png
  - 02_feature_analysis.png
  - processed_data.csv

Typical Execution Time: 2-3 minutes
""")

print("""
STEP 2: Feature Selection & Model Training (Part 2)
---------------------------------------------------
This script handles:
✓ 6 different feature selection methods
✓ Consensus-based feature selection
✓ Hyperparameter tuning (Grid/Random Search)
✓ Training 3 different models (LR, GB, XGBoost)
✓ Cross-validation with StratifiedKFold
✓ Model comparison and selection

Command:
  python lead_scoring_model_training.py

Expected Output:
  - 03_feature_selection_consensus.png
  - 04_feature_set_comparison.png
  - 05_model_comparison.png
  - best_model.pkl
  - lr_model.pkl, gb_model.pkl, xgb_model.pkl
  - optimal_features.csv
  - feature_importance.csv

Typical Execution Time: 5-10 minutes
""")

print("""
STEP 3: Model Evaluation & Implementation (Part 3)
--------------------------------------------------
This script handles:
✓ ROC curves and PR curves
✓ Calibration analysis
✓ Feature interpretation
✓ Lead score generation
✓ Segment analysis and lift calculation
✓ Business recommendations

Command:
  python lead_scoring_evaluation.py

Expected Output:
  - 06_evaluation_curves.png
  - 07_calibration_analysis.png
  - 08_feature_importance.png
  - 09_lead_scoring_results.png
  - scored_leads.csv
  - SUMMARY.txt
  - RECOMMENDATIONS.txt

Typical Execution Time: 2-3 minutes
""")

# =============================================================================
# SECTION 2: ADAPTING TO YOUR DATA
# =============================================================================
print(f"""
{'='*80}
SECTION 2: ADAPTING TO YOUR ACTUAL DATA
{'='*80}

Your current data structure should match this template:

REQUIRED COLUMNS:
─────────────────

EMAIL METRICS (High availability expected):
  - email_opens              (integer: number of email opens)
  - email_clicks             (integer: number of clicks in emails)
  - email_unsubscribe        (binary: 0/1)
  - emails_received          (integer: total emails sent)

WEBSITE METRICS (Medium availability - ~70%):
  - website_visits           (integer: number of visits)
  - pages_per_session        (float: average pages per visit)
  - time_on_site_minutes     (float: minutes spent on site)
  - return_visitor           (binary: 0/1)

WEBINAR & CONTENT (Low availability - ~30%):
  - webinars_attended        (integer: number of webinars)
  - webinar_questions_asked  (integer: questions in webinars)
  - whitepapers_downloaded   (integer: count)
  - case_studies_viewed      (integer: count)

CRM DATA:
  - existing_client          (binary: 0/1)
  - account_size_usd_millions (float: AUM or investment size)

TEMPORAL:
  - days_since_last_email    (integer: days)
  - days_since_last_website_visit (integer: days)

TARGET VARIABLE:
  - converted                (binary: 1 = engaged with sales, 0 = no engagement)

COLUMN MAPPING EXAMPLE:
─────────────────────

If your data has different column names, modify Part 1 like this:

# In lead_scoring_complete_guide.py, after loading your data:

df.rename(columns={{
    'opens': 'email_opens',
    'clicks': 'email_clicks',
    'website_sessions': 'website_visits',
    'avg_pages': 'pages_per_session',
    'aum_millions': 'account_size_usd_millions',
    'is_existing': 'existing_client',
    'conversion': 'converted'
}}, inplace=True)

HANDLING MISSING DATA:
─────────────────────

The code uses KNN imputation (best practice for correlated features).
If you have >50% missing for a column, consider:

1. REMOVE the feature (not enough signal)
2. CREATE an indicator variable (has_webinar_data = 1 if not null)
3. USE SIMPLER imputation (median/mode)

Example:
  if df['webinars_attended'].isna().sum() / len(df) > 0.5:
      df['has_webinar_data'] = (~df['webinars_attended'].isna()).astype(int)
      df['webinars_attended'] = df['webinars_attended'].fillna(0)
""")

# =============================================================================
# SECTION 3: DATA VALIDATION CHECKLIST
# =============================================================================
print(f"""
{'='*80}
SECTION 3: PRE-EXECUTION DATA VALIDATION CHECKLIST
{'='*80}

Before running the scripts, validate your data:

1. TARGET VARIABLE BALANCE
   ───────────────────────
   Check:
     df['converted'].value_counts()
   
   Acceptable ratios: 10:90 to 50:50
   If <5% positives or >95% positives: Requires special handling
   
   Code to check:
     pos_rate = df['converted'].mean()
     print(f"Positive rate: {{pos_rate:.2%}}")

2. FEATURE DATA TYPES
   ──────────────────
   Ensure numeric features are float/int:
     df[['email_opens', 'website_visits']].dtypes
   
   Convert if needed:
     df['email_opens'] = pd.to_numeric(df['email_opens'], errors='coerce')

3. OUTLIERS
   ────────
   Check for unrealistic values:
     df['time_on_site_minutes'].describe()
     
   Example issues:
     - Time on site: 999999 minutes (likely error)
     - Account size: negative values
     - Clicks > Opens (impossible)

4. MISSING VALUES
   ───────────────
   Assess missing patterns:
     print(df.isnull().sum() / len(df) * 100)
   
   Expected:
     - Email: 0-5% missing (high quality data)
     - Website: 20-40% missing (not all track this)
     - Webinar: 60-80% missing (engagement-dependent)

5. DATA VOLUME
   ───────────
   Minimum sample size:
     - 1000 records for basic model
     - 5000+ for robust model
     - 10000+ for complex model with many features
   
   Check:
     print(f"Total records: {{len(df)}}")

6. TEMPORAL CONSISTENCY
   ───────────────────
   Ensure no future data:
     df['days_since_last_email'].max()  # Should be reasonable (0-365)
   
   All timestamps should be in past:
     df['date_column'].max() <= today

7. VALUE RANGES
   ────────────
   Example validation:
     assert df['email_opens'].min() >= 0
     assert df['return_visitor'].isin([0, 1]).all()
     assert df['account_size_usd_millions'] >= 0

VALIDATION SCRIPT:
─────────────────
""")

validation_script = '''
import pandas as pd
import numpy as np

def validate_lead_scoring_data(df):
    """Comprehensive data validation"""
    issues = []
    
    # 1. Check required columns
    required = ['email_opens', 'website_visits', 'converted']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # 2. Check data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].dtype == 'object':
            issues.append(f"{col} should be numeric, got {df[col].dtype}")
    
    # 3. Check target variable
    if 'converted' in df.columns:
        unique_vals = df['converted'].unique()
        if not set(unique_vals).issubset({0, 1, np.nan}):
            issues.append(f"Target should be binary (0/1), got {unique_vals}")
    
    # 4. Check class balance
    if 'converted' in df.columns:
        pos_rate = df['converted'].mean()
        if pos_rate < 0.05 or pos_rate > 0.95:
            issues.append(f"Severe class imbalance: {pos_rate:.2%} positive")
    
    # 5. Check for negative values where shouldn't be
    non_negative = ['email_opens', 'website_visits', 'time_on_site_minutes']
    for col in non_negative:
        if col in df.columns and (df[col] < 0).any():
            issues.append(f"{col} has negative values")
    
    # 6. Check sample size
    if len(df) < 1000:
        issues.append(f"Sample size too small: {len(df)} records (need 1000+)")
    
    if issues:
        print("DATA VALIDATION ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        return False
    else:
        print("✓ All data validation checks passed!")
        return True

# Usage:
# df = pd.read_csv('your_lead_data.csv')
# validate_lead_scoring_data(df)
'''

print(validation_script)

# =============================================================================
# SECTION 4: INTERPRETING THE RESULTS
# =============================================================================
print(f"""
{'='*80}
SECTION 4: INTERPRETING MODEL RESULTS
{'='*80}

AUC-ROC INTERPRETATION:
──────────────────────
✓ 0.90-1.00: Excellent (model is highly accurate)
✓ 0.80-0.90: Good (useful for business)
✓ 0.70-0.80: Fair (acceptable with caution)
✓ 0.60-0.70: Poor (limited usefulness)
✓ 0.50-0.60: Very Poor (barely better than random)
✗ 0.50:      No predictive power

What it means:
- AUC = probability that model ranks random positive higher than random negative
- 0.80 AUC = 80% chance model correctly orders two leads

PRECISION vs RECALL:
───────────────────
Precision:  "Of the leads I mark as 'hot', how many actually convert?"
            - High precision = fewer false alarms
            - Use when follow-up cost is high

Recall:     "Of actual converters, how many do I identify as 'hot'?"
            - High recall = find most opportunities
            - Use when missing leads is costly

Trade-off:
- High precision, low recall: Conservative, miss opportunities
- Low precision, high recall: Aggressive, waste resources on false positives

For asset management: Aim for balanced precision/recall (F1-score)

FEATURE IMPORTANCE INTERPRETATION:
─────────────────────────────────

Logistic Regression Coefficients:
  Positive coefficient (+0.35): Increases conversion likelihood
  Negative coefficient (-0.15): Decreases conversion likelihood
  Larger absolute value: Stronger effect

Example interpretation:
  email_engagement_score: +0.45  → Strong positive signal
  email_unsubscribe: -0.82       → Strong negative signal
  days_since_last_email: -0.12   → Slight negative (recency matters)

Feature importance (from tree-based models):
  Value between 0 and 1
  Higher = more important for predictions
  Not directly interpretable as "effect size"

MODEL PERFORMANCE BY SEGMENT:
────────────────────────────

Hot Leads (Score > 80):
  Actual conversion rate: 65%
  Lift vs baseline: 3.2x
  Action: Assign to account manager immediately

Warm Leads (Score 60-80):
  Actual conversion rate: 35%
  Lift vs baseline: 1.7x
  Action: Include in targeted email campaigns

Lukewarm Leads (Score 40-60):
  Actual conversion rate: 18%
  Lift vs baseline: 0.9x
  Action: Monitor and nurture

What this means:
- Focusing on hot leads gives 3.2x better conversion
- 1/3 of leads marked as warm actually convert
- Baseline conversion is ~11%, hot leads achieve ~35%
""")

# =============================================================================
# SECTION 5: TROUBLESHOOTING
# =============================================================================
print(f"""
{'='*80}
SECTION 5: TROUBLESHOOTING COMMON ISSUES
{'='*80}

PROBLEM 1: Low AUC Score (< 0.65)
────────────────────────────────

Causes:
  1. Target variable is not properly defined
  2. Feature data quality is poor
  3. Missing too many values
  4. Imbalanced classes (too few conversions)

Solutions:
  a) Verify target variable
     df['converted'].value_counts()
     
  b) Check feature distributions
     df.describe()
     
  c) Reduce missing value threshold
     # Drop features with >50% missing
     missing_pct = df.isnull().sum() / len(df)
     df = df[missing_pct[missing_pct < 0.5].index]
     
  d) Increase sample size if possible
     # Model needs 5000+ records for good performance

PROBLEM 2: High AUC but Predicts All 1s or All 0s
──────────────────────────────────────────────────

Cause: Severe class imbalance

Solution: Already handled with class_weight='balanced'
  - Check classification report
  - Precision and recall should both be reasonable
  - If recall is 1.0 but precision is low: model is guessing 1 too much

PROBLEM 3: Model Works on Training Data but Fails on Test Data
──────────────────────────────────────────────────────────────

Cause: Overfitting

Check: Train AUC vs Test AUC gap
  Train AUC: 0.92
  Test AUC: 0.68
  Gap: 0.24 (too large = overfitting)

Solutions:
  1. Reduce model complexity
     - Decrease max_depth in tree models
     - Increase regularization (C parameter in LR)
     
  2. Use fewer features
     - Select top 15-20 most important features
     
  3. Use cross-validation
     - 5-fold CV score should match test AUC

PROBLEM 4: Features Don't Match Intuition
──────────────────────────────────────────

Intuition: "Email opens should be important"
Model says: "Email opens is not important"

Possible reasons:
  1. Email opens may be proxy for email volume, not quality
  2. Your best converters may not open emails (already known client)
  3. Feature engineering needed (open_rate = opens/emails_received)
  4. Confounding variable (account size is stronger signal)

Solutions:
  1. Create derived features (rates, ratios, interactions)
  2. Domain expert review of top features
  3. A/B test with sales team (do hot leads really convert?)
  4. Check correlation matrix (may be multicollinearity)

PROBLEM 5: Too Many Features Selected
──────────────────────────────────────

Issue: Model selected 50+ features, hard to interpret

Solution: Use consensus approach
  # Only use features selected by 3+ methods
  # Reduces to ~15-20 features
  # Much more stable and interpretable

Code:
  consensus_features = feature_consensus[feature_consensus['Consensus_Count'] >= 3]
  n_features = len(consensus_features)
""")

# =============================================================================
# SECTION 6: PRODUCTION DEPLOYMENT
# =============================================================================
print(f"""
{'='*80}
SECTION 6: DEPLOYING TO PRODUCTION
{'='*80}

STEP 1: Save Model and Configuration
─────────────────────────────────────

import pickle

# Save model
with open('lead_scoring_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save feature list
with open('features.txt', 'w') as f:
    f.write(','.join(optimal_features))

# Save scaler (if needed)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

STEP 2: Create Scoring Function
────────────────────────────────

def score_leads(input_df):
    '''
    Input: DataFrame with raw data (same columns as training)
    Output: DataFrame with 'lead_score' column (0-100)
    '''
    
    # 1. Load model
    with open('lead_scoring_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 2. Load features
    with open('features.txt', 'r') as f:
        features = f.read().split(',')
    
    # 3. Preprocess (same as training)
    X_input = preprocess_data(input_df)[features]
    
    # 4. Get predictions
    probs = model.predict_proba(X_input)[:, 1]
    
    # 5. Scale to 0-100
    scores = (probs * 100).round(0).astype(int)
    
    return scores

STEP 3: Set Up Daily Batch Scoring
───────────────────────────────────

# batch_scoring.py
import pandas as pd
from datetime import datetime

def daily_scoring_job():
    '''Runs every day at 6 AM'''
    
    # 1. Extract new leads from CRM (last 24 hours)
    new_leads = extract_from_crm(days=1)
    
    # 2. Score them
    scores = score_leads(new_leads)
    new_leads['lead_score'] = scores
    
    # 3. Save to database
    save_to_database(new_leads)
    
    # 4. Log metrics
    log_metrics({{
        'leads_scored': len(new_leads),
        'avg_score': scores.mean(),
        'timestamp': datetime.now()
    }})
    
    # 5. Alert on hot leads
    hot_leads = new_leads[new_leads['lead_score'] > 80]
    if len(hot_leads) > 0:
        send_alert(f"{{len(hot_leads)}} hot leads found!")

# Schedule with cron or task scheduler
# 0 6 * * * python batch_scoring.py

STEP 4: Monitor Model Performance
─────────────────────────────────

# monitoring.py
import pandas as pd

def monitor_model():
    '''Monthly model monitoring'''
    
    # Get predictions from last month
    predictions = get_predictions_from_db(days=30)
    
    # Calculate actual conversions
    actuals = get_actuals_from_crm(days=30)
    
    # Compare
    auc = roc_auc_score(actuals, predictions)
    
    # Check for drift
    if auc < 0.75:
        print("WARNING: Model performance degraded!")
        trigger_retraining()
    
    # Monthly report
    generate_report(auc, actuals, predictions)

STEP 5: Integration with CRM/Tools
──────────────────────────────────

Option A: Salesforce
  - Use Salesforce Flow to call Python API
  - Update Lead Score field daily
  
Option B: HubSpot
  - Use Custom Property for Lead Score
  - Integrate via HubSpot API

Option C: Excel/Spreadsheet
  - Export daily scores to CSV
  - Upload to shared drive
  
Option D: Custom Dashboard
  - Create Tableau/Power BI dashboard
  - Shows leads by score segment
  - Updates daily from API

EXAMPLE CRM INTEGRATION (Python):
─────────────────────────────────

from salesforce_api import Salesforce

def update_salesforce_leads():
    sf = Salesforce(api_key='...')
    
    # Get all leads
    leads = sf.query("SELECT Id, Name, Email FROM Lead WHERE Status='Open'")
    
    # Score them
    scores = score_leads(leads)
    
    # Update in Salesforce
    for lead, score in zip(leads, scores):
        sf.update_record(
            object_type='Lead',
            record_id=lead['Id'],
            data={{
                'Custom_Score__c': score,
                'Priority__c': 'High' if score > 70 else 'Medium'
            }}
        )
    
    print(f"Updated {{len(leads)}} leads in Salesforce")
""")

# =============================================================================
# SECTION 7: KEY PARAMETERS & CONFIGURATION
# =============================================================================
print(f"""
{'='*80}
SECTION 7: KEY PARAMETERS & CONFIGURATION GUIDE
{'='*80}

LEAD SCORE THRESHOLDS (Customize for your business):
──────────────────────────────────────────────────────

Default Configuration:
  Hot Lead:      Score >= 80  → Assign to Account Manager
  Warm Lead:     60-79        → Include in email campaign
  Lukewarm:      40-59        → Monitor
  Cool Lead:     20-39        → Nurture
  Cold Lead:     < 20         → Low priority

If your sales team is small:
  Hot Lead:      >= 85
  Warm Lead:     70-84
  (Fewer leads, higher quality)

If you need more opportunities:
  Hot Lead:      >= 70
  Warm Lead:     50-69
  (More leads, may have lower conversion)

FEATURE ENGINEERING PARAMETERS:
───────────────────────────────

Email Engagement Score weights (can adjust):
  0.4 × Opens/Max_Opens + 0.5 × Clicks/Max_Clicks + 0.1 × Emails_Received

Website Engagement Score weights:
  0.4 × Visits/Max_Visits + 0.3 × Pages/Max_Pages + 
  0.2 × Time/Max_Time + 0.1 × Return_Visitor

Change if needed based on domain knowledge

MISSING VALUE IMPUTATION:
─────────────────────────

Current: KNN with k=5 neighbors
Alternative options:

For highly correlated features:
  from sklearn.impute import SimpleImputer
  imputer = SimpleImputer(strategy='median')  # Fast, simple

For complex patterns:
  from sklearn.experimental import enable_iterative_imputer
  from sklearn.impute import IterativeImputer
  imputer = IterativeImputer()  # More sophisticated

MODEL HYPERPARAMETERS (Can fine-tune):
──────────────────────────────────────

Logistic Regression:
  C: [0.001, 0.01, 0.1, 1, 10]
  ← Increase C if underfitting (low train AUC)
  ← Decrease C if overfitting (train >> test AUC)
  
  penalty: 'l1' vs 'l2'
  ← Use 'l1' for feature selection (drives some to 0)
  ← Use 'l2' for balanced penalties

Gradient Boosting:
  n_estimators: [100, 200, 300]
  ← Increase for better accuracy (slower)
  
  learning_rate: [0.01, 0.05, 0.1]
  ← Decrease for finer tuning (slower)
  
  max_depth: [3, 5, 7]
  ← Decrease if overfitting
  ← Increase if underfitting

XGBoost:
  Similar to Gradient Boosting with:
  scale_pos_weight = n_negative / n_positive
  ← Automatically handles class imbalance
""")

# =============================================================================
# SECTION 8: RUNNING THE FULL PIPELINE
# =============================================================================
print(f"""
{'='*80}
SECTION 8: COMPLETE EXECUTION - START HERE
{'='*80}

QUICK START (3 steps):
──────────────────────

1. PREPARE YOUR DATA
   
   Save your lead data as CSV:
     your_lead_data.csv
   
   With columns:
     email_opens, email_clicks, emails_received, website_visits,
     pages_per_session, time_on_site_minutes, return_visitor,
     webinars_attended, whitepapers_downloaded, case_studies_viewed,
     existing_client, account_size_usd_millions, days_since_last_email,
     days_since_last_website_visit, converted
   
   Check data quality:
     python validate_data.py your_lead_data.csv

2. UPDATE THE DATA LOADING CODE
   
   In lead_scoring_complete_guide.py, replace:
   
     df_raw = generate_sample_data(n_samples=5000)
   
   With:
   
     df_raw = pd.read_csv('your_lead_data.csv')

3. RUN COMPLETE PIPELINE
   
   # Run all three scripts in sequence
   python lead_scoring_complete_guide.py
   python lead_scoring_model_training.py
   python lead_scoring_evaluation.py

WHAT YOU'LL GET:
────────────────
✓ Best performing model (saved as best_model.pkl)
✓ 4 visualization reports (PNG files)
✓ Feature importance scores
✓ Lead scores for all leads
✓ Business recommendations
✓ Performance metrics and interpretation

TOTAL TIME: 15-20 minutes for full execution

NEXT STEPS:
───────────
1. Review visualizations (PNG files)
2. Read RECOMMENDATIONS.txt
3. Validate with your sales team
4. Deploy to production
5. Set up daily scoring
6. Monitor and retrain monthly

COMMON MISTAKES TO AVOID:
────────────────────────

❌ Using future data (leads that haven't happened yet)
✓ Use data from complete periods (full months/quarters)

❌ Data leakage (using post-conversion information)
✓ Only use features collected BEFORE conversion event

❌ Ignoring class imbalance
✓ Use class_weight='balanced' (already in code)

❌ Not splitting train/test properly
✓ Use stratified split (already in code)

❌ Overfitting to training data
✓ Monitor train vs test AUC gap (< 0.1 is good)

❌ Deploying without validation
✓ Test on recent holdout data
✓ Get sales team feedback

❌ Never updating the model
✓ Retrain monthly with new data
✓ Monitor for performance drift

SUPPORT & RESOURCES:
───────────────────

Scikit-learn Documentation:
  https://scikit-learn.org/stable/

Feature Selection Guide:
  https://scikit-learn.org/stable/modules/feature_selection.html

Model Evaluation:
  https://scikit-learn.org/stable/modules/model_evaluation.html

XGBoost Docs:
  https://xgboost.readthedocs.io/

Questions to ask yourself:
  1. Does model performance meet business requirements? (AUC > 0.75?)
  2. Do top features align with sales team experience?
  3. Is lift analysis realistic? (3x for hot leads?)
  4. Can this be operationalized with current tools?
  5. What's the ROI? (Sales cycles reduced by X%, conversion +Y%?)

{'='*80}
""")

print("""
✓ GUIDE COMPLETE!

You now have:
1. Complete Python code for the entire pipeline
2. Step-by-step instructions
3. Troubleshooting guide
4. Production deployment guide
5. Monitoring and retraining procedures

Next step: Run the scripts with your actual data!
""")

