import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import joblib

df = pd.read_csv('../AI Job Market Dataset.csv')

print("=== ROZKŁAD SALARY WG WSZYSTKICH KATEGORII ===")
for col in ['job_title', 'experience_level', 'company_size', 'education_level',
            'remote_type', 'hiring_urgency', 'company_industry']:
    group_std = df.groupby(col)['salary'].mean().std()
    print(f"{col:<25} rozrzut średnich: {group_std:.0f} USD")

print("\n=== PERMUTATION IMPORTANCE (prawdziwa ważność) ===")
data = joblib.load('../preprocesing/processed_data.joblib')
model = joblib.load('../salary_model_rf.joblib')
preprocessor = joblib.load('../preprocesing/preprocessor.joblib')

# Permutation importance jest bardziej wiarygodna niż feature_importances_
perm = permutation_importance(model, data['X_test'], data['y_test'],
                               n_repeats=10, random_state=42, n_jobs=-1)

feature_names = preprocessor.get_feature_names_out()
perm_df = pd.DataFrame({
    'feature': feature_names,
    'importance': perm.importances_mean
}).sort_values('importance', ascending=False).head(15)

print(perm_df.to_string(index=False))