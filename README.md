# Przewidywanie wynagrodzeń w branży IT (Artificial Intelligence)

## O projekcie
Celem projektu jest analiza czynników wpływających na wysokość wynagrodzenia w sektorze Artificial Intelligence oraz budowa modelu regresyjnego szacującego roczne zarobki. Zbiór danych zawiera informacje o ofertach pracy z całego świata, uwzględniając wymogi techniczne, lokalizację, poziom stanowiska oraz profil pracodawcy.

## Zbiór danych
Dane użyte w projekcie pochodzą z serwisu Kaggle:
[AI and Data Science Job Market Dataset 2020-2026](https://www.kaggle.com/datasets/shree0910/ai-and-data-science-job-market-dataset-20202026)

* **Liczba instancji (rekordów):** 10 345
* **Liczba atrybutów:** 19 (w tym zmienna docelowa)

### Atrybuty
1. **job_id** - unikalny identyfikator oferty pracy
2. **job_title** - nazwa stanowiska (np. Data Scientist, AI Engineer)
3. **company_size** - wielkość przedsiębiorstwa (Startup, Medium, itp.)
4. **company_industry** - branża firmy (np. Technology, Finance)
5. **country** - kraj zatrudnienia
6. **remote_type** - tryb pracy (Remote, Hybrid, Onsite)
7. **experience_level** - poziom stanowiska (Entry, Mid, Senior)
8. **years_experience** - wymagana liczba lat doświadczenia
9. **education_level** - wymagane wykształcenie (Bachelor, Master, PhD)
10. **skills_python** - wymóg znajomości języka Python (1 - tak, 0 - nie)
11. **skills_sql** - wymóg znajomości SQL (1 - tak, 0 - nie)
12. **skills_ml** - wymóg znajomości Machine Learning (1 - tak, 0 - nie)
13. **skills_deep_learning** - wymóg znajomości Deep Learning (1 - tak, 0 - nie)
14. **skills_cloud** - wymóg znajomości chmury (1 - tak, 0 - nie)
15. **job_posting_month** - miesiąc publikacji ogłoszenia
16. **job_posting_year** - rok publikacji ogłoszenia
17. **hiring_urgency** - pilność zatrudnienia (Low, Medium, High)
18. **job_openings** - liczba dostępnych wakatów w ramach ogłoszenia
19. **salary** - roczne wynagrodzenie w USD (**zmienna docelowa**)

## Typ problemu i modelowanie
Projekt rozwiązuje **problem regresji**, gdzie zmienną docelową jest przewidywanie wartości ciągłej `salary` (rocznego wynagrodzenia).

Do trenowania i ewaluacji zostaną użyte dwa algorytmy:
1. **Linear Regression (Regresja Liniowa)** – prosty model liniowy, który posłuży jako punkt odniesienia (baseline).
2. **Random Forest Regressor (Model Lasu Losowego)** – nieliniowy model zespołowy, który powinien lepiej uchwycić skomplikowane zależności między zmiennymi a wysokością wynagrodzenia.

