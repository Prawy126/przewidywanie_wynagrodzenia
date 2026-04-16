import matplotlib.pyplot as plt
import numpy as np

from uczenie.main import model
from test import preprocessor

# Pobieramy ważność cech z modelu
importances = model.feature_importances_
# Pobieramy nazwy kolumn po transformacji (OneHotEncoder tworzy ich dużo)
feature_names = preprocessor.get_feature_names_out()

# Sortujemy cechy od najważniejszej
indices = np.argsort(importances)[-10:]  # bierzemy 10 najważniejszych

# Konfiguracja stylu
plt.style.use('seaborn-v0_8-darkgrid')  # lub 'default', 'ggplot', 'bmh'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 9

# Tworzenie wykresu
fig, ax = plt.subplots(figsize=(12, 8))

# Kolory - gradient od najsłabszej do najsilniejszej
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))

# Tworzenie wykresu słupkowego
bars = ax.barh(range(len(indices)), importances[indices],
               align="center", color=colors, edgecolor='black', linewidth=0.7)

# Dodanie wartości na końcu każdego słupka
for i, (idx, bar) in enumerate(zip(indices, bars)):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2,
            f'{importances[idx]:.4f}',
            ha='left', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Skrócenie długich nazw cech (opcjonalne)
short_names = []
for name in [feature_names[i] for i in indices]:
    if len(name) > 40:
        short_names.append(name[:37] + '...')
    else:
        short_names.append(name)

ax.set_yticks(range(len(indices)))
ax.set_yticklabels(short_names)
ax.set_xlabel("Względna ważność", fontweight='bold')
ax.set_title("Top 10 czynników wpływających na wynagrodzenie",
             fontweight='bold', pad=20)

# Siatka dla lepszej czytelności
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Dodanie marginesów
plt.tight_layout()

# Zapisz PRZED show()
plt.savefig("wykresy/top_10_czynnikow_wplywajacych_na_wynagrodzenie.png",
            dpi=300, bbox_inches='tight')

# Wyświetl
plt.show()