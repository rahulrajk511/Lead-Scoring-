"""
=============================================================================
LEAD SCORING MODEL FOR ASSET MANAGEMENT
Complete End-to-End Data Science Pipeline
=============================================================================

Problem Statement:
- Determine relative weights of digital engagement metrics (opens, clicks, 
  website visits, webinars, etc.)
- Binary target: 1 = engaged with sales (converted), 0 = no engagement
- Challenge: High email fill rate but low fill rates for other channels
- Goal: Accurate lead scoring with interpretable feature weights

Author: Data Science Team
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
    learning_curve
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    LogisticRegression
)
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
    recall_score
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('/home/claude/lead_scoring_outputs', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
pd.set_option('display.max_columns', None)

print("="*80)
print("LEAD SCORING MODEL - COMPLETE PIPELINE")
print("="*80)

# =============================================================================
# SECTION 1: SYNTHETIC DATA GENERATION (Replace with your actual data)
# =============================================================================
print("\n[1] GENERATING SAMPLE DATA")
print("-" * 80)

def generate_sample_data(n_samples=5000):
    """
    Generate synthetic lead scoring data matching asset management context
    
    Features:
    - Email engagement: opens, clicks, unsubscribes
    - Website: page visits, pages per session, time on site, return visits
    - Content: webinars attended, whitepapers downloaded, case studies viewed
    - CRM data: existing client, account size
    - Temporal: recency of last engagement
    """
    
    np.random.seed(42)
    
    # Create base conversion probability
    conversion_prob = np.random.rand(n_samples)
    
    data = {
        # EMAIL METRICS (High availability)
        'email_opens': np.where(
            conversion_prob > 0.4,
            np.random.randint(1, 20, n_samples),
            np.random.randint(0, 8, n_samples)
        ),
        'email_clicks': np.where(
            conversion_prob > 0.45,
            np.random.randint(0, 10, n_samples),
            np.random.randint(0, 3, n_samples)
        ),
        'email_unsubscribe': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'emails_received': np.random.randint(5, 50, n_samples),
        
        # WEBSITE METRICS (Medium availability - 70%)
        'website_visits': np.where(
            np.random.rand(n_samples) > 0.3,
            np.where(conversion_prob > 0.3, 
                    np.random.randint(2, 15, n_samples),
                    np.random.randint(0, 5, n_samples)),
            np.nan
        ),
        'pages_per_session': np.where(
            np.random.rand(n_samples) > 0.3,
            np.where(conversion_prob > 0.3, 
                    np.random.uniform(3, 15, n_samples),
                    np.random.uniform(1, 5, n_samples)),
            np.nan
        ),
        'time_on_site_minutes': np.where(
            np.random.rand(n_samples) > 0.3,
            np.where(conversion_prob > 0.3, 
                    np.random.uniform(5, 60, n_samples),
                    np.random.uniform(0.5, 10, n_samples)),
            np.nan
        ),
        'return_visitor': np.where(
            np.random.rand(n_samples) > 0.3,
            np.where(conversion_prob > 0.35, 
                    np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                    np.random.choice([0, 1], n_samples, p=[0.8, 0.2])),
            np.nan
        ),
        
        # WEBINAR METRICS (Low availability - 30%)
        'webinars_attended': np.where(
            np.random.rand(n_samples) > 0.7,
            np.where(conversion_prob > 0.4, 
                    np.random.randint(1, 5, n_samples),
                    0),
            np.nan
        ),
        'webinar_questions_asked': np.where(
            np.random.rand(n_samples) > 0.8,
            np.random.choice([0, 1, 2], n_samples),
            np.nan
        ),
        
        # CONTENT METRICS (Low availability - 35%)
        'whitepapers_downloaded': np.where(
            np.random.rand(n_samples) > 0.65,
            np.where(conversion_prob > 0.4, 
                    np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3]),
                    np.random.choice([0, 1], n_samples, p=[0.9, 0.1])),
            np.nan
        ),
        'case_studies_viewed': np.where(
            np.random.rand(n_samples) > 0.65,
            np.where(conversion_prob > 0.4, 
                    np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.3, 0.25, 0.15]),
                    np.random.choice([0, 1], n_samples, p=[0.85, 0.15])),
            np.nan
        ),
        
        # CRM METRICS
        'existing_client': np.where(conversion_prob > 0.7, 
                                   np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                                   np.random.choice([0, 1], n_samples, p=[0.9, 0.1])),
        'account_size_usd_millions': np.where(
            conversion_prob > 0.6,
            np.random.exponential(50, n_samples),
            np.random.exponential(10, n_samples)
        ),
        
        # TEMPORAL METRICS
        'days_since_last_email': np.random.randint(0, 90, n_samples),
        'days_since_last_website_visit': np.where(
            np.random.rand(n_samples) > 0.3,
            np.random.randint(0, 60, n_samples),
            np.nan
        ),
        
        # TARGET VARIABLE
        'converted': np.where(conversion_prob > 0.65, 1, 0)
    }
    
    df = pd.DataFrame(data)
    return df

df_raw = generate_sample_data(n_samples=5000)
print(f"✓ Generated {len(df_raw)} sample leads")
print(f"✓ Features: {df_raw.shape[1]-1} (excluding target)")
print(f"✓ Target distribution:\n{df_raw['converted'].value_counts(normalize=True)}\n")
print(f"✓ Data shape: {df_raw.shape}")
print("\nFirst few rows:")
print(df_raw.head(10))

# =============================================================================
# SECTION 2: DATA PROFILING & MISSING VALUE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("[2] DATA PROFILING & MISSING VALUE ANALYSIS")
print("-" * 80)

# Detailed missing value analysis
missing_analysis = pd.DataFrame({
    'Feature': df_raw.columns,
    'Missing_Count': df_raw.isnull().sum(),
    'Missing_Percentage': (df_raw.isnull().sum() / len(df_raw) * 100).round(2),
    'Data_Type': df_raw.dtypes,
    'Non_Null_Mean': df_raw.mean(),
    'Non_Null_Std': df_raw.std()
})

print("\nMissing Value Analysis:")
print(missing_analysis.to_string())

# Visualize missing data
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Missing value bar chart
missing_pct = (df_raw.isnull().sum() / len(df_raw) * 100).sort_values(ascending=False)
missing_pct[missing_pct > 0].plot(kind='barh', ax=axes[0], color='coral')
axes[0].set_xlabel('Missing Percentage (%)')
axes[0].set_title('Missing Data by Feature')
axes[0].axvline(x=50, color='red', linestyle='--', label='50% threshold')
axes[0].legend()

# Target distribution
df_raw['converted'].value_counts().plot(kind='bar', ax=axes[1], color=['skyblue', 'salmon'])
axes[1].set_title('Target Distribution (Class Imbalance)')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(['Not Converted (0)', 'Converted (1)'], rotation=0)

plt.tight_layout()
plt.savefig('/home/claude/lead_scoring_outputs/01_missing_data_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: missing data visualization")

# =============================================================================
# SECTION 3: DATA PREPROCESSING - STRATEGIC APPROACH
# =============================================================================
print("\n" + "="*80)
print("[3] DATA PREPROCESSING - STRATEGIC APPROACH")
print("-" * 80)

df = df_raw.copy()

print("\n3.1: HANDLING MISSING VALUES")
print("-" * 40)

# Strategy: Different handling for different missing patterns
# Problem: High email fill rate (0% missing) vs low rates for other channels

# For website metrics (30% missing):
# Use KNN imputation based on behavior similarity
from sklearn.impute import KNNImputer, SimpleImputer

website_features = ['website_visits', 'pages_per_session', 'time_on_site_minutes', 'return_visitor']
webinar_features = ['webinars_attended', 'webinar_questions_asked']
content_features = ['whitepapers_downloaded', 'case_studies_viewed']
temporal_features = ['days_since_last_website_visit']

# Strategy 1: KNN Imputation for website & engagement metrics
# This preserves relationships between features
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
df[website_features] = knn_imputer.fit_transform(df[website_features])

# Strategy 2: Create "has_data" indicator features BEFORE imputation
# This captures the information that channel wasn't used
df['has_webinar_activity'] = (~df_raw['webinars_attended'].isnull()).astype(int)
df['has_content_activity'] = (~df_raw[content_features].isnull()).any(axis=1).astype(int)
df['has_website_activity'] = (~df_raw[website_features].isnull()).any(axis=1).astype(int)

print(f"✓ Created 'has_X_activity' indicator features")

# Fill webinar and content with 0 (meaning no activity)
df[webinar_features] = df[webinar_features].fillna(0)
df[content_features] = df[content_features].fillna(0)
df['days_since_last_website_visit'] = df['days_since_last_website_visit'].fillna(
    df['days_since_last_website_visit'].median()
)

print(f"✓ Missing values after imputation: {df.isnull().sum().sum()}")

print("\n3.2: FEATURE ENGINEERING - CREATE ENGAGEMENT SCORES")
print("-" * 40)

# Create aggregated engagement metrics that capture multi-channel behavior
# These will help with the low fill rate problem by combining available signals

# Email Engagement Score (normalized)
df['email_engagement_score'] = (
    (df['email_opens'] / df['email_opens'].max()) * 0.4 +
    (df['email_clicks'] / df['email_clicks'].max()) * 0.5 +
    ((df['emails_received'] - df['email_opens'].min()) / 
     (df['emails_received'].max() - df['emails_received'].min())) * 0.1
)

# Website Engagement Score
df['website_engagement_score'] = (
    (df['website_visits'] / df['website_visits'].max()) * 0.4 +
    (df['pages_per_session'] / df['pages_per_session'].max()) * 0.3 +
    (df['time_on_site_minutes'] / df['time_on_site_minutes'].max()) * 0.2 +
    df['return_visitor'] * 0.1
)

# Content/Webinar Engagement Score
df['content_engagement_score'] = (
    (df['webinars_attended'] / (df['webinars_attended'].max() + 1)) * 0.4 +
    (df['whitepapers_downloaded'] / (df['whitepapers_downloaded'].max() + 1)) * 0.3 +
    (df['case_studies_viewed'] / (df['case_studies_viewed'].max() + 1)) * 0.3
)

# Overall Engagement Score (multi-channel signal)
df['overall_engagement_score'] = (
    df['email_engagement_score'] * 0.5 +
    df['website_engagement_score'] * 0.3 +
    df['content_engagement_score'] * 0.2
)

# Recency Score (normalize and invert - recent is better)
df['recency_score'] = 1 - (df['days_since_last_email'] / df['days_since_last_email'].max())

print("✓ Created engagement score features:")
print("  - email_engagement_score")
print("  - website_engagement_score")
print("  - content_engagement_score")
print("  - overall_engagement_score")
print("  - recency_score")

print("\n3.3: OUTLIER DETECTION & TREATMENT")
print("-" * 40)

# Use IQR method for outlier detection
def treat_outliers_iqr(data, columns, multiplier=1.5):
    df_clean = data.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outlier_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        
        # Cap outliers instead of removing (preserve information)
        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        if outlier_count > 0:
            print(f"  ✓ {col}: {outlier_count} outliers capped")
    
    return df_clean

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns.remove('converted')  # Don't treat target variable

df = treat_outliers_iqr(df, numeric_columns, multiplier=1.5)

print("\n3.4: FEATURE NORMALIZATION")
print("-" * 40)

# Use RobustScaler (better for outliers) instead of StandardScaler
# Separate scaling for features with different patterns
scaler = RobustScaler()

features_to_scale = [col for col in df.columns if col not in ['converted']]
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

print(f"✓ Scaled {len(features_to_scale)} features using RobustScaler")
print(f"  (RobustScaler is preferred for data with outliers)")

print("\n3.5: FEATURE SELECTION - OVERVIEW")
print("-" * 40)
print("""
We will use MULTIPLE feature selection methods:
1. Statistical Tests (F-test, Mutual Information)
2. Model-based (Random Forest Feature Importance)
3. Recursive Feature Elimination (RFE)
4. Sequential Forward Selection (SFS)
5. L1 Regularization (Logistic Regression Coefficients)

This multi-method approach provides robust feature selection.
""")

# =============================================================================
# SECTION 4: EXPLORATORY FEATURE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("[4] EXPLORATORY FEATURE ANALYSIS")
print("-" * 80)

# Correlation with target
correlations = pd.DataFrame({
    'Feature': df.columns[:-1],
    'Correlation_with_Target': [df[col].corr(df['converted']) for col in df.columns[:-1]]
}).sort_values('Correlation_with_Target', key=abs, ascending=False)

print("\nCorrelation with Conversion (Top 15):")
print(correlations.head(15).to_string(index=False))

# Statistical tests
print("\n\nStatistical Tests (F-statistic for ANOVA):")
f_scores, p_values = f_classif(df.drop('converted', axis=1), df['converted'])
f_stat_df = pd.DataFrame({
    'Feature': df.columns[:-1],
    'F_Statistic': f_scores,
    'P_Value': p_values
}).sort_values('F_Statistic', ascending=False)

print(f_stat_df.head(15).to_string(index=False))

# Mutual Information (captures non-linear relationships)
print("\n\nMutual Information Scores:")
mi_scores = mutual_info_classif(df.drop('converted', axis=1), df['converted'], random_state=42)
mi_df = pd.DataFrame({
    'Feature': df.columns[:-1],
    'Mutual_Information': mi_scores
}).sort_values('Mutual_Information', ascending=False)

print(mi_df.head(15).to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Correlation heatmap (top features only)
top_features = correlations.head(10)['Feature'].tolist() + ['converted']
sns.heatmap(df[top_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0, 0])
axes[0, 0].set_title('Correlation Matrix (Top Features)')

# F-statistics
f_stat_df.head(15).plot(x='Feature', y='F_Statistic', kind='barh', ax=axes[0, 1], legend=False)
axes[0, 1].set_title('F-Statistic Scores (ANOVA)')
axes[0, 1].set_xlabel('F-Statistic')

# Mutual Information
mi_df.head(15).plot(x='Feature', y='Mutual_Information', kind='barh', ax=axes[1, 0], legend=False)
axes[1, 0].set_title('Mutual Information Scores')
axes[1, 0].set_xlabel('Mutual Information')

# Feature importance preview (Random Forest)
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_temp.fit(df.drop('converted', axis=1), df['converted'])
rf_importance = pd.DataFrame({
    'Feature': df.columns[:-1],
    'Importance': rf_temp.feature_importances_
}).sort_values('Importance', ascending=False)

rf_importance.head(15).plot(x='Feature', y='Importance', kind='barh', ax=axes[1, 1], legend=False)
axes[1, 1].set_title('Random Forest Feature Importance (Preview)')
axes[1, 1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('/home/claude/lead_scoring_outputs/02_feature_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: feature analysis visualization")

# Save processed data
df.to_csv('/home/claude/lead_scoring_outputs/processed_data.csv', index=False)
print("✓ Saved: processed_data.csv")

# Return for next section
processed_df = df.copy()
print("\n✓ Data preprocessing completed successfully!")

