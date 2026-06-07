"""
Train XGBoost classifier for cardiovascular risk prediction.
Data source: BRFSS 2023 (Behavioral Risk Factor Surveillance System).

Features (16):
    Demographics:   Age, Sex
    Body:           BMI Category
    Lifestyle:      Smoking, Heavy Alcohol, Exercise Frequency
    Conditions:     High Blood Pressure, High Cholesterol, Diabetes, Stroke,
                    Other Cancer, COPD, Kidney Disease, Depression
    Self-rated:     Physical Health Days, Mental Health Days

Target: _MICHD (heart attack or coronary heart disease)
Unknown / Don't know / Refused -> NaN (XGBoost handles missing natively)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("Loading BRFSS 2023 dataset (only the columns we need)...")
# BRFSS 2023 ships with lowercase column names
cols_needed_lower = [
    '_michd',       # Target
    '_ageg5yr',     # Age category
    'sexvar',       # Sex
    '_bmi5cat',     # BMI category
    'smoke100',     # Ever smoked 100+ cigarettes
    '_rfdrhv8',     # Heavy alcohol consumption (calculated)
    'exerany2',     # Any exercise (Yes/No)
    'exeroft1',     # Exercise frequency (encoded)
    '_rfhype6',     # High blood pressure (calculated)
    '_rfchol3',     # High cholesterol (calculated)
    'diabete4',     # Diabetes
    'cvdstrk3',     # Stroke
    'chcocnc1',     # Other cancer (excludes skin)
    'chccopd3',     # COPD
    'chckdny2',     # Kidney disease
    'addepev3',     # Depression
    'menthlth',     # Days of poor mental health
    'physhlth',     # Days of poor physical health (replaces GENHLTH)
]
df = pd.read_csv('BRFSS2023.csv', usecols=cols_needed_lower, low_memory=False)
df.columns = [c.upper() for c in df.columns]

# Drop rows where the target is missing
df = df.dropna(subset=['_MICHD'])

# ============== TARGET ==============
df['HeartDisease'] = df['_MICHD'].replace({2.0: 0.0, 1.0: 1.0})

# ============== AGE ==============
df = df[df['_AGEG5YR'].between(1, 13)]
df['Age'] = df['_AGEG5YR']

# ============== SEX (Male=1, Female=0) ==============
df['Sex'] = df['SEXVAR'].replace({1.0: 1.0, 2.0: 0.0})

# ============== BMI CATEGORY ==============
df['BMICategory'] = df['_BMI5CAT']

# ============== SMOKING ==============
df['Smoking'] = df['SMOKE100'].replace(
    {1.0: 1.0, 2.0: 0.0, 7.0: np.nan, 9.0: np.nan}
)

# ============== HEAVY ALCOHOL ==============
df['Alcohol'] = df['_RFDRHV8'].replace(
    {1.0: 0.0, 2.0: 1.0, 9.0: np.nan}
)

# ============== EXERCISE FREQUENCY (decoded to times-per-month) ==============
def decode_exercise(row):
    any_ex = row['EXERANY2']
    freq = row['EXEROFT1']
    if any_ex == 2.0:
        return 0.0
    if any_ex in (7.0, 9.0) or pd.isna(any_ex):
        return np.nan
    if pd.isna(freq) or freq in (777.0, 999.0):
        return np.nan
    if freq == 888.0:
        return 0.0
    if 101.0 <= freq <= 199.0:
        return (freq - 100.0) * 4.33
    if 201.0 <= freq <= 299.0:
        return freq - 200.0
    return np.nan

df['ExerciseFreq'] = df.apply(decode_exercise, axis=1)

# ============== HIGH BLOOD PRESSURE ==============
df['HighBP'] = df['_RFHYPE6'].replace(
    {1.0: 0.0, 2.0: 1.0, 9.0: np.nan}
)

# ============== HIGH CHOLESTEROL ==============
df['HighChol'] = df['_RFCHOL3'].replace(
    {1.0: 0.0, 2.0: 1.0, 9.0: np.nan}
)

# ============== DIABETES (3-level) ==============
df['Diabetes'] = df['DIABETE4'].replace(
    {1.0: 2.0, 2.0: 1.0, 4.0: 1.0, 3.0: 0.0, 7.0: np.nan, 9.0: np.nan}
)

# ============== STROKE ==============
df['Stroke'] = df['CVDSTRK3'].replace(
    {1.0: 1.0, 2.0: 0.0, 7.0: np.nan, 9.0: np.nan}
)

# ============== OTHER CANCER ==============
df['OtherCancer'] = df['CHCOCNC1'].replace(
    {1.0: 1.0, 2.0: 0.0, 7.0: np.nan, 9.0: np.nan}
)

# ============== COPD ==============
df['COPD'] = df['CHCCOPD3'].replace(
    {1.0: 1.0, 2.0: 0.0, 7.0: np.nan, 9.0: np.nan}
)

# ============== KIDNEY DISEASE ==============
df['KidneyDisease'] = df['CHCKDNY2'].replace(
    {1.0: 1.0, 2.0: 0.0, 7.0: np.nan, 9.0: np.nan}
)

# ============== DEPRESSION ==============
df['Depression'] = df['ADDEPEV3'].replace(
    {1.0: 1.0, 2.0: 0.0, 7.0: np.nan, 9.0: np.nan}
)

# ============== MENTAL HEALTH DAYS ==============
df['MentalHealth'] = df['MENTHLTH'].replace(
    {88.0: 0.0, 77.0: np.nan, 99.0: np.nan}
)

# ============== PHYSICAL HEALTH DAYS (replaces General Health) ==============
df['PhysicalHealth'] = df['PHYSHLTH'].replace(
    {88.0: 0.0, 77.0: np.nan, 99.0: np.nan}
)

# Final feature set (16 features) -- GeneralHealth -> PhysicalHealth
FEATURES = [
    'Age', 'Sex', 'BMICategory', 'Smoking', 'Alcohol', 'ExerciseFreq',
    'HighBP', 'HighChol', 'Diabetes', 'Stroke',
    'OtherCancer', 'COPD', 'KidneyDisease', 'Depression',
    'MentalHealth', 'PhysicalHealth',
]
X = df[FEATURES].astype(float)
y = df['HeartDisease'].astype(int)

print(f"\nDataset size: {len(df):,} rows")
print(f"Heart disease prevalence: {y.mean()*100:.2f}%")
print(f"\nMissing (Unknown) values per feature:")
for col in FEATURES:
    n_missing = X[col].isna().sum()
    pct = n_missing / len(X) * 100
    print(f"  {col:<16s} {n_missing:>7,} ({pct:5.2f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nscale_pos_weight: {scale:.2f}")

print("\nTraining XGBoost Classifier (16 features)...")
model = xgb.XGBClassifier(
    scale_pos_weight=scale,
    random_state=42,
    eval_metric='logloss',
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"\n{'='*60}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"{'='*60}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=['No Heart Disease', 'Heart Disease'],
                            digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nTop feature importances:")
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in importances.items():
    bar = '#' * int(imp * 100)
    print(f"  {feat:<16s} {imp:.4f}  {bar}")

with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved to xgboost_model.pkl")
