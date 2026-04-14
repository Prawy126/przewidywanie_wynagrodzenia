import joblib
import pandas as pd
import numpy as np

# 1. Wczytujemy model i preprocesor
model = joblib.load('../salary_model.joblib')
preprocessor = joblib.load('../preprocesing/preprocessor.joblib')

# 2. Tworzymy dane nowego pracownika
new_data = pd.DataFrame([{
    'job_title': 'AI Engineer',
    'company_size': 'MNC',
    'company_industry': 'Technology',
    'experience_level': 'Senior',
    'years_experience': 8,
    'education_level': 'Master',
    'country': 'Germany',
    'remote_type': 'Remote',
    'job_openings': 5,
    'job_posting_month': 4,
    'job_posting_year': 2026,
    'hiring_urgency': 'Medium',
    'skills_python': 1,
    'skills_sql': 1,
    'skills_ml': 1,
    'skills_deep_learning': 1,
    'skills_cloud': 0
}])

# 3. Transformacja cykliczna miesiąca - tak samo jak w preprocessingu!
new_data['month_sin'] = np.sin(2 * np.pi * new_data['job_posting_month'] / 12)
new_data['month_cos'] = np.cos(2 * np.pi * new_data['job_posting_month'] / 12)
new_data = new_data.drop('job_posting_month', axis=1)

# 4. Przetwarzamy i przewidujemy
new_data_processed = preprocessor.transform(new_data)
prediction = model.predict(new_data_processed)

print(f"Przewidywane wynagrodzenie: {prediction[0]:.2f} USD")