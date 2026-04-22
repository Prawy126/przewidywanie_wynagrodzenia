import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Upewnienie się, że folder istnieje
os.makedirs('wykresy', exist_ok=True)

# Wczytanie danych
df = pd.read_csv('AI Job Market Dataset.csv')

# Ustawienie stylu wykresów
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ---------------------------------------------------------
# 1. Średnie wynagrodzenie a Poziom Doświadczenia (Boxplot)
# ---------------------------------------------------------
plt.figure()
# Używamy wbudowanej funkcji pandas do boxplotów
ax = df.boxplot(column='salary', by='experience_level', grid=True, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
plt.title('Rozkład wynagrodzeń wg poziomu doświadczenia', pad=20)
plt.suptitle('') # Usunięcie domyślnego tytułu nakładającego się z pandas
plt.xlabel('Poziom doświadczenia')
plt.ylabel('Wynagrodzenie (USD)')
plt.tight_layout()
plt.savefig('wykresy/wynagrodzenie_vs_doswiadczenie.png', dpi=300, bbox_inches='tight')
plt.close('all')

# ---------------------------------------------------------
# 2. Najlepiej płatne stanowiska w AI
# ---------------------------------------------------------
plt.figure()
top_jobs = df.groupby('job_title')['salary'].mean().sort_values(ascending=True).tail(10)
top_jobs.plot(kind='barh', color='skyblue', edgecolor='black')
plt.title('Top 10 najlepiej płatnych stanowisk (Średnia)')
plt.xlabel('Średnie wynagrodzenie (USD)')
plt.ylabel('Stanowisko')
plt.tight_layout()
plt.savefig('wykresy/top_stanowiska_zarobki.png', dpi=300, bbox_inches='tight')
plt.close('all')

# ---------------------------------------------------------
# 3. Wpływ pracy zdalnej na pensję
# ---------------------------------------------------------
plt.figure()
remote_salary = df.groupby('remote_type')['salary'].mean().sort_values(ascending=False)
remote_salary.plot(kind='bar', color=['#4daf4a', '#377eb8', '#e41a1c'], edgecolor='black')
plt.title('Średnie wynagrodzenie a tryb pracy')
plt.xlabel('Tryb pracy')
plt.ylabel('Średnie wynagrodzenie (USD)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('wykresy/wplyw_pracy_zdalnej.png', dpi=300, bbox_inches='tight')
plt.close('all')

# ---------------------------------------------------------
# 4. Premia za umiejętności
# ---------------------------------------------------------
plt.figure()
skills_cols = [col for col in df.columns if col.startswith('skills_')]
skill_premiums = {}

for skill in skills_cols:
    # Średnia pensja w ofertach, gdzie dana umiejętność jest wymagana (wartość 1)
    mean_with_skill = df[df[skill] == 1]['salary'].mean()
    if not np.isnan(mean_with_skill):
        clean_name = skill.replace('skills_', '').replace('_', ' ').title()
        skill_premiums[clean_name] = mean_with_skill

# Sortowanie i wybór top 10 najlepiej płatnych umiejętności
skill_premiums_s = pd.Series(skill_premiums).sort_values(ascending=True).tail(10)
skill_premiums_s.plot(kind='barh', color='coral', edgecolor='black')
plt.title('Najbardziej opłacalne umiejętności\n(Średnia dla ofert z wymogiem)', pad=15)
plt.xlabel('Średnie wynagrodzenie (USD)')
plt.ylabel('Umiejętność')
plt.tight_layout()
plt.savefig('wykresy/premia_za_umiejetnosci.png', dpi=300, bbox_inches='tight')
plt.close('all')

# ---------------------------------------------------------
# 5. Wynagrodzenia na świecie (Top 10 najczęstszych krajów)
# ---------------------------------------------------------
plt.figure()
# Najpierw znajdujemy 10 krajów z największą liczbą ofert, aby uniknąć anomalii z pojedynczych ofert
top_countries = df['country'].value_counts().head(10).index
country_salary = df[df['country'].isin(top_countries)].groupby('country')['salary'].mean().sort_values(ascending=True)

country_salary.plot(kind='barh', color='mediumpurple', edgecolor='black')
plt.title('Średnie wynagrodzenie w 10 najpopularniejszych krajach', pad=15)
plt.xlabel('Średnie wynagrodzenie (USD)')
plt.ylabel('Kraj')
plt.tight_layout()
plt.savefig('wykresy/zarobki_wg_kraju.png', dpi=300, bbox_inches='tight')
plt.close('all')

print("Sukces! Wygenerowano 5 wykresów i zapisano w folderze 'wykresy/'.")