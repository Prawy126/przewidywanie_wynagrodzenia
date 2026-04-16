import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# 1. Wczytujemy dane
df = pd.read_csv('../AI Job Market Dataset.csv')

# 2. Sprawdzamy braki danych
numeric_features = ['years_experience', 'job_openings', 'job_posting_year']
categorical_features = [
    'job_title', 'company_size', 'company_industry', 'country',
    'remote_type', 'experience_level', 'education_level', 'hiring_urgency'
]
binary_features = [col for col in df.columns if col.startswith('skills_')]

missing = df[numeric_features].isnull().sum()
if missing.any():
    print(f"Uwaga! Braki danych w kolumnach numerycznych:\n{missing[missing > 0]}")
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# 3. Transformacja cykliczna dla miesięcy
df['month_sin'] = np.sin(2 * np.pi * df['job_posting_month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['job_posting_month'] / 12)

# Dodajemy nowe kolumny cykliczne do numeric_features, usuwamy oryginalny miesiąc
cyclic_features = ['month_sin', 'month_cos']

# 4. Oddzielamy cel i usuwamy niepotrzebne kolumny
X = df.drop(['job_id', 'salary', 'job_posting_month'], axis=1)
y = df['salary']

# 5. Podział na zbiór treningowy i testowy PRZED transformacją
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Tworzymy procesor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features + cyclic_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('bin', 'passthrough', binary_features)
    ])

# 7. fit tylko na danych treningowych, transform na obu zbiorach
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)  # tylko transform, nie fit!

# 8. Zapisujemy wyniki
joblib.dump({
    'X_train': X_train_processed,
    'X_test': X_test_processed,
    'y_train': y_train,
    'y_test': y_test
}, 'processed_data.joblib')

joblib.dump(preprocessor, 'preprocessor.joblib')

print(f"Sukces! Dane przetworzone.")
print(f"Kształt zbioru treningowego: {X_train_processed.shape}")
print(f"Kształt zbioru testowego:    {X_test_processed.shape}")