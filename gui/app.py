import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# Konfiguracja strony
st.set_page_config(page_title="Przewidywanie Wynagrodzenia AI", layout="wide")

# Funkcja do wczytywania zasobów
@st.cache_resource
def load_assets():
    # Ścieżki relatywne do głównego folderu projektu
    model_rf = joblib.load('uczenie/salary_model_rf.joblib')
    model_lr = joblib.load('uczenie/salary_model_lr.joblib')
    preprocessor = joblib.load('preprocesing/preprocessor.joblib')
    df = pd.read_csv('AI Job Market Dataset.csv')
    return model_rf, model_lr, preprocessor, df

try:
    model_rf, model_lr, preprocessor, df = load_assets()
except Exception as e:
    st.error(f"Błąd podczas ładowania modeli lub danych: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("Nawigacja")
page = st.sidebar.radio("Wybierz sekcję:", ["Statystyki i Analiza", "Kalkulator Wynagrodzenia"])

# --- SEKCOJA 1: STATYSTYKI ---
if page == "Statystyki i Analiza":
    st.title("📊 Analiza Rynku Pracy AI")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Liczba ofert", len(df))
    col2.metric("Średnie wynagrodzenie", f"${df['salary'].mean():,.0f}")
    col3.metric("Mediana wynagrodzenia", f"${df['salary'].median():,.0f}")

    st.subheader("Rozkład wynagrodzeń")
    fig, ax = plt.subplots(figsize=(10, 4))
    df['salary'].hist(bins=30, ax=ax, color='#4e79a7', edgecolor='black')
    ax.set_xlabel("Wynagrodzenie (USD)")
    ax.set_ylabel("Liczba ofert")
    st.pyplot(fig)

    st.subheader("Top 10 czynników wpływających na płacę")
    if hasattr(model_rf, 'feature_importances_'):
        importances = model_rf.feature_importances_
        feature_names = preprocessor.get_feature_names_out()
        indices = np.argsort(importances)[-10:]
        
        # Oczyszczanie nazw cech (usuwanie przedrostków cat__, num__ itp.)
        clean_names = [feature_names[i].split('__')[-1].replace('_', ' ').title() for i in indices]
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        # Tworzenie poprawnego gradientu kolorów
        colors = plt.cm.viridis(np.linspace(0.4, 0.8, len(indices)))
        
        ax2.barh(range(len(indices)), importances[indices], color=colors)
        ax2.set_yticks(range(len(indices)))
        ax2.set_yticklabels(clean_names)
        ax2.set_xlabel("Względna ważność")
        st.pyplot(fig2)

# --- SEKCJA 2: KALKULATOR ---
else:
    st.title("💰 Kalkulator Wynagrodzenia")
    st.write("Wprowadź swoje dane, aby otrzymać szacunkowe wynagrodzenie.")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        
        with c1:
            job_title = st.selectbox("Stanowisko", df['job_title'].unique())
            experience_level = st.selectbox("Poziom doświadczenia", df['experience_level'].unique())
            years_experience = st.slider("Lata doświadczenia", 0, 30, 5)
            country = st.selectbox("Kraj", df['country'].unique())
            education_level = st.selectbox("Wykształcenie", df['education_level'].unique())

        with c2:
            company_size = st.selectbox("Wielkość firmy", df['company_size'].unique())
            company_industry = st.selectbox("Branża", df['company_industry'].unique())
            remote_type = st.selectbox("Tryb pracy", df['remote_type'].unique())
            hiring_urgency = st.selectbox("Pilność rekrutacji", df['hiring_urgency'].unique())
            job_openings = st.number_input("Liczba otwartych stanowisk", 1, 100, 5)

        st.write("### Umiejętności")
        skill_cols = st.columns(3)
        skills = {}
        binary_features = [col for col in df.columns if col.startswith('skills_')]
        
        for i, skill in enumerate(binary_features):
            col_idx = i % 3
            clean_name = skill.replace('skills_', '').replace('_', ' ').title()
            skills[skill] = skill_cols[col_idx].checkbox(clean_name)

        model_choice = st.radio("Model predykcyjny", ["Random Forest (dokładniejszy)", "Regresja Liniowa"], horizontal=True)
        
        submit = st.form_submit_button("Oblicz wynagrodzenie")

    if submit:
        # Przygotowanie danych do modelu
        input_data = {
            'job_title': job_title,
            'company_size': company_size,
            'company_industry': company_industry,
            'experience_level': experience_level,
            'years_experience': years_experience,
            'education_level': education_level,
            'country': country,
            'remote_type': remote_type,
            'job_openings': job_openings,
            'job_posting_month': 4, # domyślnie obecny miesiąc
            'job_posting_year': 2026,
            'hiring_urgency': hiring_urgency,
        }
        # Dodanie skilli
        for s, val in skills.items():
            input_data[s] = 1 if val else 0
            
        input_df = pd.DataFrame([input_data])
        
        # Transformacja cykliczna miesiąca
        input_df['month_sin'] = np.sin(2 * np.pi * input_df['job_posting_month'] / 12)
        input_df['month_cos'] = np.cos(2 * np.pi * input_df['job_posting_month'] / 12)
        input_df = input_df.drop('job_posting_month', axis=1)

        # Preprocessing
        processed_input = preprocessor.transform(input_df)
        
        # Predykcja
        selected_model = model_rf if "Random Forest" in model_choice else model_lr
        prediction = selected_model.predict(processed_input)[0]
        
        # Obliczanie widełek (na podstawie MAE z README.md)
        mae = 2543 if "Random Forest" in model_choice else 2447
        lower_bound = max(0, prediction - mae) # Nie chcemy pensji ujemnej
        upper_bound = prediction + mae
        
        st.success(f"### Szacowane wynagrodzenie: ${prediction:,.0f} USD")
        st.info(f"**Przewidywane widełki płacowe:** ${lower_bound:,.0f} — ${upper_bound:,.0f} USD")
        st.write("Pamiętaj, że widełki są obliczane na podstawie średniego błędu modelu (MAE).")
