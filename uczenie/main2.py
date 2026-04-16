import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Wczytujemy oryginalny dataset
df = pd.read_csv('../AI Job Market Dataset.csv')

# 2. Transformacja cykliczna miesięcy
df['month_sin'] = np.sin(2 * np.pi * df['job_posting_month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['job_posting_month'] / 12)

X_full = df.drop(['job_id', 'salary', 'job_posting_month'], axis=1)
X_reduced = df.drop(['job_id', 'salary', 'job_posting_month',
                     'job_title', 'experience_level'], axis=1)  # <-- bez dominujących
y = df['salary']

# 3. Budujemy preprocessory dla obu wariantów
def build_preprocessor(X):
    categorical = ['company_size', 'company_industry', 'country',
                   'remote_type', 'education_level', 'hiring_urgency']
    # job_title i experience_level dodajemy tylko jeśli są w X
    categorical = [c for c in categorical if c in X.columns]
    if 'job_title' in X.columns:
        categorical.append('job_title')
    if 'experience_level' in X.columns:
        categorical.append('experience_level')

    numeric = [c for c in ['years_experience', 'job_openings',
                            'job_posting_year', 'month_sin', 'month_cos']
               if c in X.columns]
    binary = [c for c in X.columns if c.startswith('skills_')]

    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
        ('bin', 'passthrough', binary)
    ])

# 4. Funkcja trenująca i ewaluująca
def train_and_evaluate(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p  = preprocessor.transform(X_test)

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train_p, y_train)

    cv_scores = cross_val_score(model, X_train_p, y_train, cv=5, scoring='r2', n_jobs=-1)
    predictions = model.predict(X_test_p)

    mae  = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2   = r2_score(y_test, predictions)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"Liczba cech po transformacji: {X_train_p.shape[1]}")
    print(f"Cross-walidacja R² (5-fold):  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Średni błąd  (MAE):           {mae:.2f} USD")
    print(f"Błąd RMSE:                    {rmse:.2f} USD")
    print(f"Dokładność (R²):              {r2:.3f}")

    return r2, mae, rmse

# 5. Trenujemy oba warianty
r2_full,    mae_full,    rmse_full    = train_and_evaluate(X_full,    y, "PEŁNY MODEL (wszystkie cechy)")
r2_reduced, mae_reduced, rmse_reduced = train_and_evaluate(X_reduced, y, "OKROJONY MODEL (bez job_title i experience_level)")

# 6. Porównanie
print(f"\n{'='*55}")
print(f"  WPŁYW USUNIĘCIA DOMINUJĄCYCH CECH")
print(f"{'='*55}")
print(f"{'Metryka':<10} {'Pełny':>12} {'Okrojony':>12} {'Różnica':>12}")
print(f"{'-'*46}")
print(f"{'R²':<10} {r2_full:>12.3f} {r2_reduced:>12.3f} {r2_reduced - r2_full:>+12.3f}")
print(f"{'MAE':<10} {mae_full:>12.2f} {mae_reduced:>12.2f} {mae_reduced - mae_full:>+12.2f} USD")
print(f"{'RMSE':<10} {rmse_full:>12.2f} {rmse_reduced:>12.2f} {rmse_reduced - rmse_full:>+12.2f} USD")