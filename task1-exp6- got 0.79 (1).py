import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# --- STEP 1: DATA LOADING ---
df = pd.read_csv(r'C:\Users\gsudh\Documents\depaul\courses\capstone\week4\task1_gecs_classification_final.csv')
df_blanks = pd.read_csv(r'C:\Users\gsudh\Documents\depaul\courses\capstone\week4\Empty cell info morningstar_gecs_industries.csv')

# --- STEP 2: KNOWLEDGE-BASED IMPUTATION ---
df['MstarGlobal'] = df['MstarGlobal'].astype(str)
df_blanks['GECS_Code'] = df_blanks['GECS_Code'].astype(str)
definition_map = dict(zip(df_blanks['GECS_Code'], df_blanks['Description']))

# Filling empty segment descriptions with official PDF definitions
df['SegmentDescription'] = df['SegmentDescription'].fillna(df['MstarGlobal'].map(definition_map))

# --- STEP 3: FEATURE ENGINEERING ---
df['master_text'] = (df['SegmentName'].fillna('') + " " + 
                     df['LongProfile'].fillna('') + " " + 
                     df['SegmentDescription'].fillna(''))

# Numerical Feature Preparation
df['log_total_rev'] = np.log1p(df['total_revenue_company_as_of'].clip(lower=0))
# Ensure the boolean column is numeric (0/1)
df['is_largest_share_segment'] = df['is_largest_share_segment'].astype(int)

le = LabelEncoder()
y = le.fit_transform(df['MstarGlobal'])

# Target columns for the hybrid model
num_cols = ['log_total_rev', 'Revenue', 'revenue_share', 'is_largest_share_segment']
X = df[['master_text'] + num_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- STEP 4: ENHANCED HYBRID PIPELINE ---
business_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + \
                      ['company', 'incorporated', 'limited', 'services', 'solutions', 'operations', 'segment']

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(
            max_features=10000, 
            stop_words=business_stop_words, 
            ngram_range=(1, 3),
            use_idf=True, 
            smooth_idf=True
        ), 'master_text'),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ]), num_cols)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LinearSVC(class_weight='balanced', random_state=42, max_iter=3000, C=0.8))
])

# --- STEP 5: TRAINING & EVALUATION ---
print("\n[SYSTEM] Training model for 145 GECS industries...")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

report = classification_report(y_test, y_pred, target_names=le.classes_.astype(str), output_dict=True)
report_df = pd.DataFrame(report).transpose()

# --- DISPLAY METRICS ---
print("\n" + "="*60)
print("GLOBAL MODEL SCORECARD")
print("="*60)
print(f"OVERALL ACCURACY:      {report_df.loc['accuracy', 'f1-score']:.4f}")
print(f"MACRO AVG F1-SCORE:    {report_df.loc['macro avg', 'f1-score']:.4f}  <-- GOAL: 0.75")
print(f"WEIGHTED AVG F1-SCORE: {report_df.loc['weighted avg', 'f1-score']:.4f}")
print("="*60)

# --- STEP 6: INDUSTRY LOOKUP PREP ---
try:
    df_lookup = pd.read_csv(r'C:\Users\gsudh\Documents\depaul\courses\capstone\week3\Empty cell info morningstar_gecs_industries.csv')
    df_lookup.columns = df_lookup.columns.str.strip()
    industry_name_map = dict(zip(df_lookup['GECS_Code'].astype(str), df_lookup['Industry_Name']))
except Exception:
    industry_name_map = {}

# --- STEP 7: INTERACTIVE "MORNINGSTAR TERMINAL" UI ---
def run_interactive_prediction():
    title = "MORNINGSTAR GECS INTELLIGENCE ENGINE"
    ui_width = 58
    
    print("\n" + "╔" + "═" * (ui_width + 2) + "╗")
    print(f"║ {title.center(ui_width)} ║")
    print("╚" + "═" * (ui_width + 2) + "╝")
    print("Type 'exit' in Segment Name to quit.\n")

    while True:
        print("─" * (ui_width + 4))
        name = input("▶ SEGMENT NAME: ")
        if name.lower() == 'exit': break
        
        profile = input("▶ LONG PROFILE: ")
        description = input("▶ DESCRIPTION:  ")
        
        try:
            total_rev = float(input("▶ TOTAL COMPANY REV ($): "))
            rev_share = float(input("▶ REVENUE SHARE (0.0-1.0): "))
            is_large = input("▶ IS LARGEST SEGMENT? (y/n): ").lower() == 'y'
            
            # Derived fields for the model
            segment_rev = total_rev * rev_share
            log_rev = np.log1p(max(0, total_rev))
            is_large_val = 1 if is_large else 0
            
        except ValueError:
            print("!! ERROR: Please enter valid numeric values.")
            continue

        # Prepare manual input
        master_text = f"{name} {profile} {description}"
        custom_input = pd.DataFrame({
            'master_text': [master_text],
            'log_total_rev': [log_rev],
            'Revenue': [segment_rev],
            'revenue_share': [rev_share],
            'is_largest_share_segment': [is_large_val]
        })
        
        # Predict
        pred_num = pipeline.predict(custom_input)
        gecs_code = le.inverse_transform(pred_num)[0]
        industry_name = industry_name_map.get(str(gecs_code), "UNKNOWN INDUSTRY")
        
        # Confidence
        decision_scores = pipeline.decision_function(custom_input)
        confidence = np.max(decision_scores)

        # UI Results Box (Fixed Alignment)
        box_w = 52
        print("\n" + "┌" + "─" * (box_w - 2) + "┐")
        print(f"│ {('GECS CODE: ' + str(gecs_code)):<{box_w-4}} │")
        print(f"│ {('INDUSTRY:  ' + industry_name.upper()):<{box_w-4}} │")
        print(f"│ {('CONFIDENCE: ' + f'{confidence:.2f}'):<{box_w-4}} │")
        print("└" + "─" * (box_w - 2) + "┘\n")

if __name__ == "__main__":
    run_interactive_prediction()