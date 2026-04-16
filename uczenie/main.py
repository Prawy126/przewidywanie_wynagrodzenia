import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Wczytujemy gotowe dane
data = joblib.load('../preprocesing/processed_data.joblib')
X_train = data['X_train']
X_test  = data['X_test']
y_train = data['y_train']
y_test  = data['y_test']

# 2. Definiujemy modele
models = {
    'Random Forest':     RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    'Regresja Liniowa':  LinearRegression(n_jobs=-1)
}

# 3. Trenujemy, ewaluujemy i zapisujemy każdy model
results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    # Trening
    model.fit(X_train, y_train)

    # Cross-walidacja na zbiorze treningowym
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print(f"Cross-walidacja R² (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Ewaluacja na zbiorze testowym
    predictions = model.predict(X_test)
    mae  = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2   = r2_score(y_test, predictions)

    print(f"Średni błąd  (MAE):  {mae:.2f} USD")
    print(f"Błąd RMSE:           {rmse:.2f} USD")
    print(f"Dokładność (R²):     {r2:.3f}")

    # Zapamiętujemy wyniki do porównania
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'CV_mean': cv_scores.mean()}

    # Zapisujemy model Z KOMPRESJĄ
    # compress=3 to kompresja zlib - idealna dla GitHub (mały plik, szybki odczyt)
    filename = 'salary_model_rf.joblib' if 'Forest' in name else 'salary_model_lr.joblib'
    joblib.dump(model, filename, compress=3)
    print(f"Model zapisany jako: {filename} (skompresowany)")

# 4. Podsumowanie porównawcze
print(f"\n{'='*50}")
print(f"  PORÓWNANIE MODELI")
print(f"{'='*50}")
print(f"{'Metryka':<20} {'Random Forest':>15} {'Regresja Liniowa':>17}")
print(f"{'-'*52}")
for metric in ['MAE', 'RMSE', 'R2', 'CV_mean']:
    rf_val = results['Random Forest'][metric]
    lr_val = results['Regresja Liniowa'][metric]
    print(f"{metric:<20} {rf_val:.3f:>15} {lr_val:.3f:>17}")

# 5. Wskazujemy lepszy model
best = max(results, key=lambda x: results[x]['R2'])
print(f"\nLepszy model wg R²: {best}")