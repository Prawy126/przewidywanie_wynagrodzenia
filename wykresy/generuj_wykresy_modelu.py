import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

# Wczytanie danych i modeli
data = joblib.load('preprocesing/processed_data.joblib')
model_rf = joblib.load('uczenie/salary_model_rf.joblib')
model_lr = joblib.load('uczenie/salary_model_lr.joblib')

X_test = data['X_test']
y_test = data['y_test']

# Słownik modeli dla pętli
models = {
    'rf': ('Random Forest', model_rf, 'royalblue', 'indigo'),
    'lr': ('Regresja Liniowa', model_lr, 'mediumseagreen', 'darkgreen')
}

# Ustawienie stylu
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)

for model_key, (model_name, model, color_scatter, color_hist) in models.items():
    # Predykcje
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Wykres 1: Rzeczywiste vs Przewidywane
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.3, color=color_scatter, edgecolor='none')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'Rzeczywiste vs Przewidywane Wynagrodzenie ({model_name})', pad=20)
    plt.xlabel('Rzeczywiste Wynagrodzenie (USD)')
    plt.ylabel('Przewidywane Wynagrodzenie (USD)')
    plt.tight_layout()
    plt.savefig(f'wykresy/rzeczywiste_vs_przewidywane_{model_key}.png', dpi=300)
    plt.close()

    # Wykres 2: Rozkład błędów (Reszt)
    plt.figure()
    plt.hist(residuals, bins=50, color=color_hist, edgecolor='black', alpha=0.7)
    plt.title(f'Rozkład Błędów (Reszt) - {model_name}', pad=20)
    plt.xlabel('Błąd (Rzeczywiste - Przewidywane) USD')
    plt.ylabel('Liczba obserwacji')
    plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero (Brak błędu)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'wykresy/rozklad_bledow_{model_key}.png', dpi=300)
    plt.close()

print("Wygenerowano wykresy ewaluacyjne dla obu modeli (Random Forest i Regresja Liniowa).")
