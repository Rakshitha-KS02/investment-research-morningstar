import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# ==============================================================================
# --- STEP 1: PATHS & DATA LOADING ---
# ==============================================================================
print("[SYSTEM] Initializing GECS Integrated Classification Framework...")

# Paths for Week 4 and Week 5 directories
path_w4 = Path(r"C:\Users\gsudh\Documents\depaul\courses\capstone\week4")
path_w5 = Path(r"C:\Users\gsudh\Documents\depaul\courses\capstone\week5")

try:
    # Load Task 1 Data
    df_t1 = pd.read_csv(path_w4 / "task1_gecs_classification_final.csv")
    df_lookup = pd.read_csv(path_w4 / "Empty cell info morningstar_gecs_industries.csv")
    
    # Load Task 2 Data
    df_t2_main = pd.read_excel(path_w5 / "task2_subindustry_classification_final.xlsx")
    df_map = pd.read_excel(path_w5 / "GECS_Activities2026.xlsx")

    # Standardize headers (strip whitespace)
    for d in [df_t1, df_lookup, df_t2_main, df_map]:
        d.columns = d.columns.str.strip()
        
    print("[SYSTEM] All datasets loaded successfully.")

except Exception as e:
    print(f"CRITICAL ERROR: Data initialization failed.\n{e}")
    sys.exit()

# ==============================================================================
# --- STEP 2: DYNAMIC COLUMN DETECTION & CLEAN MAPPING ---
# ==============================================================================
print("[SYSTEM] Detecting columns and building Mapping Dictionaries...")

def find_col(df, keyword):
    """Detects column names regardless of spaces or underscores."""
    for col in df.columns:
        if keyword.lower() in col.lower().replace(" ", "_"):
            return col
    return None

# Detect headers for Task 2 Hierarchical Mapping
col_s_id = find_col(df_map, 'Sector_Id')
col_s_nm = find_col(df_map, 'Sector_name')
col_g_id = find_col(df_map, 'group_Id')
col_g_nm = find_col(df_map, 'group_name')
col_i_id = find_col(df_map, 'Industry_ID')
col_i_nm = find_col(df_map, 'Industry_name')
col_a_id = find_col(df_map, 'Activity_ID')
col_a_nm = find_col(df_map, 'Activity_name')
col_a_df = find_col(df_map, 'Activity_Definition')

def force_clean_id(val):
    """Converts Excel scientific/float IDs (1.01E+09) to clean 10-digit strings."""
    try:
        if pd.isna(val): return None
        return str(int(float(val))).strip()
    except:
        return str(val).strip()

def build_clean_mapping(df, id_col, name_col):
    if not id_col or not name_col: return {}
    mapping = df[[id_col, name_col]].dropna(subset=[id_col]).copy()
    mapping[id_col] = mapping[id_col].apply(force_clean_id)
    mapping[name_col] = mapping[name_col].astype(str).str.strip()
    return mapping.drop_duplicates(subset=[id_col]).set_index(id_col)[name_col].to_dict()

# Build Task 1 Lookup Maps
industry_name_map_t1 = dict(zip(df_lookup['GECS_Code'].astype(str), df_lookup['Industry_Name']))
definition_map_t1 = dict(zip(df_lookup['GECS_Code'].astype(str), df_lookup['Description'].fillna('No official definition.')))

# Build Task 2 Maps (Sector, Group, Industry)
sector_map = build_clean_mapping(df_map, col_s_id, col_s_nm)
group_map  = build_clean_mapping(df_map, col_g_id, col_g_nm)
industry_map = build_clean_mapping(df_map, col_i_id, col_i_nm)

# Build Activity Map (Name and Definition)
activity_map = {}
if col_a_id and col_a_nm:
    act_df = df_map[[col_a_id, col_a_nm, col_a_df]].dropna(subset=[col_a_id]).copy()
    act_df[col_a_id] = act_df[col_a_id].apply(force_clean_id)
    for _, row in act_df.drop_duplicates(subset=[col_a_id]).iterrows():
        activity_map[row[col_a_id]] = {
            'name': str(row[col_a_nm]).strip().upper(),
            'definition': str(row[col_a_df]).strip()
        }

# ==============================================================================
# --- STEP 3: FEATURE ENGINEERING & DATA SPLITTING ---
# ==============================================================================
print("[SYSTEM] Processing Features...")

# Task 1: Fill missing descriptions using knowledge mapping
df_t1['MstarGlobal'] = df_t1['MstarGlobal'].astype(str)
df_t1['SegmentDescription'] = df_t1['SegmentDescription'].fillna(df_t1['MstarGlobal'].map(definition_map_t1))

df_t1['master_text'] = (
    df_t1['SegmentName'].fillna('') + " " + 
    df_t1['LongProfile'].fillna('') + " " + 
    df_t1['SegmentDescription'].fillna('')
).str.lower()

df_t1['log_rev_transformed'] = np.log1p(df_t1['total_revenue_company_as_of'].clip(lower=0))

le_t1 = LabelEncoder()
y_t1 = le_t1.fit_transform(df_t1['MstarGlobal'])
X_t1 = df_t1[['master_text', 'log_rev_transformed', 'Revenue', 'revenue_share']]

# Task 2: Integrate Revenue and prepare training text
df_t2 = pd.merge(df_t2_main, df_t1[['CompanyId', 'Revenue']], on='CompanyId', how='left')
df_t2['training_text'] = (
    df_t2['SegmentName'].fillna('').astype(str) + " " + 
    df_t2['SegmentDescription'].fillna('').astype(str) + " " + 
    df_t2['Industry'].fillna('').astype(str)
).str.lower()

# Minimum sample filter for stratification
counts_t2 = df_t2['SubIndustry'].value_counts()
df_t2_filtered = df_t2[df_t2['SubIndustry'].isin(counts_t2[counts_t2 >= 3].index)].copy()

# Perform Train/Test Splits
X_train_t1, X_test_t1, y_train_t1, y_test_t1 = train_test_split(
    X_t1, y_t1, test_size=0.2, stratify=y_t1, random_state=42
)

X_train_t2, X_test_t2, y_train_t2, y_test_t2 = train_test_split(
    df_t2_filtered['training_text'], 
    df_t2_filtered['SubIndustry'], 
    test_size=0.2, 
    stratify=df_t2_filtered['SubIndustry'], 
    random_state=42
)

# ==============================================================================
# --- STEP 4: MODEL PIPELINES & TRAINING ---
# ==============================================================================
print("[SYSTEM] Training Multi-Task Pipelines...")

business_stops = list(TfidfVectorizer(stop_words='english').get_stop_words()) + \
                 ['company', 'incorporated', 'limited', 'services', 'solutions', 'operations', 'segment']

pipeline_t1 = Pipeline([
    ('prep', ColumnTransformer([
        ('text', TfidfVectorizer(max_features=10000, stop_words=business_stops, ngram_range=(1, 3)), 'master_text'),
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='constant', fill_value=0)),
            ('scl', StandardScaler())
        ]), ['log_rev_transformed', 'Revenue', 'revenue_share'])
    ])),
    ('clf', LinearSVC(class_weight='balanced', random_state=42, C=0.8, max_iter=3000))
])

pipeline_t2 = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=40000, stop_words='english')),
    ('clf', LinearSVC(class_weight='balanced', random_state=42, max_iter=7000))
])

pipeline_t1.fit(X_train_t1, y_train_t1)
pipeline_t2.fit(X_train_t2, y_train_t2)

# ==============================================================================
# --- STEP 5: PERFORMANCE ANALYTICS ---
# ==============================================================================
rep1 = classification_report(y_test_t1, pipeline_t1.predict(X_test_t1), target_names=le_t1.classes_.astype(str), output_dict=True)
rep2 = classification_report(y_test_t2, pipeline_t2.predict(X_test_t2), output_dict=True, zero_division=0)

print("\n" + "="*85)
print("INTEGRATED PERFORMANCE ANALYTICS")
print("="*85)
print(f"{'METRIC':<30} | {'TASK 1 (MACRO)':<18} | {'TASK 2 (SUB)':<18}")
print("-" * 85)
print(f"{'Macro Avg F1-Score':<30} | {rep1['macro avg']['f1-score']:<18.4f} | {rep2['macro avg']['f1-score']:<18.4f}")
print(f"{'Weighted Avg F1-Score':<30} | {rep1['weighted avg']['f1-score']:<18.4f} | {rep2['weighted avg']['f1-score']:<18.4f}")
print(f"{'Overall Accuracy':<30} | {rep1['accuracy']:<18.4f} | {rep2['accuracy']:<18.4f}")
print("-" * 85)
print(f"{'GLOBAL SYSTEM SYSTEM F1:':<30} {(rep1['macro avg']['f1-score'] + rep2['macro avg']['f1-score'])/2:.4f}")
print("="*85 + "\n")

# ==============================================================================
# --- STEP 6: UNIFIED INTERACTIVE ENGINE ---
# ==============================================================================
def run_integrated_engine():
    print("╔" + "═"*75 + "╗")
    print("║" + " "*14 + "UNIFIED GECS MACRO & HIERARCHICAL RECO ENGINE" + " "*16 + "║")
    print("╚" + "═"*75 + "╝")
    print("Type 'exit' to quit.\n")

    while True:
        print("─"*77)
        desc = input("▶ ENTER NEW DESCRIPTION: ")
        if desc.lower() == 'exit': break
        
        try:
            revenue = float(input("▶ ENTER REVENUE ($): "))
        except ValueError:
            print("!! ERROR: Please enter a valid numeric revenue value.")
            continue

        # --- TASK 1: CALIBRATED MACRO PREDICTION ---
        input_t1 = pd.DataFrame({
            'master_text': [desc.lower()],
            'log_rev_transformed': [np.log1p(max(0, revenue))],
            'Revenue': [revenue],
            'revenue_share': [1.0]
        })
        
        scores = pipeline_t1.decision_function(input_t1)[0]
        T = 0.12 
        scaled_scores = np.clip(scores / T, -20, 20)
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        probs = (exp_scores / np.sum(exp_scores)) * 100
        top3_indices = np.argsort(scores)[-3:][::-1]

        print("\n┌" + "─"*74 + "┐")
        print(f"│ {'RANK':<6} | {'GECS CODE':<12} | {'TASK 1 MACRO INDUSTRY NAME':<35} | {'CERTAINTY %':<12} │")
        print("├" + "─"*74 + "┤")
        for i, idx in enumerate(top3_indices):
            code = str(le_t1.classes_[idx])
            name = industry_name_map_t1.get(code, "UNKNOWN").upper()
            rank_tag = f"#{i+1}{' *' if i == 0 else '  '}"
            print(f"│ {rank_tag:<6} | {code:<12} | {name[:33]:<35} | {probs[idx]:>11.1f}% │")
        print("└" + "─"*74 + "┘")

        # --- TASK 2: HIERARCHICAL MAPPING ---
        pred_id_t2 = str(pipeline_t2.predict([desc.lower()])[0]).strip().replace('.0', '')
        s_id, g_id, i_id = pred_id_t2[:3], pred_id_t2[:5], pred_id_t2[:8]
        act_info = activity_map.get(pred_id_t2, {'name': 'NOT FOUND', 'definition': 'No definition available.'})

        print("┌" + "─"*74 + "┐")
        print(f"│ {'LEVEL':<15} | {'GECS CODE':<20} | {'MAPPED VALUE NAME':<33} │")
        print("├" + "─"*74 + "┤")
        print(f"│ {'SECTOR':<15} | {s_id:<20} | {str(sector_map.get(s_id, 'NOT FOUND'))[:33]:<33} │")
        print(f"│ {'INDUSTRY GROUP':<15} | {g_id:<20} | {str(group_map.get(g_id, 'NOT FOUND'))[:33]:<33} │")
        print(f"│ {'INDUSTRY':<15} | {i_id:<20} | {str(industry_map.get(i_id, 'NOT FOUND'))[:33]:<33} │")
        print(f"│ {'SUB-ACTIVITY':<15} | {pred_id_t2:<20} | {str(act_info['name'])[:33]:<33} │")
        print("├" + "─"*74 + "┤")
        print(f"│ {'DEFINITION:':<74} │")
        
        raw_def = act_info['definition']
        wrapped_lines = [raw_def[i:i+70] for i in range(0, len(raw_def), 70)]
        for line in wrapped_lines:
            print(f"│   {line:<71} │")
        print("└" + "─"*74 + "┘\n")

if __name__ == "__main__":
    run_integrated_engine()