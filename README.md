# Przewidywanie wynagrodzeń w branży IT (Artificial Intelligence)

## O projekcie
Celem projektu jest analiza czynników wpływających na wysokość wynagrodzenia w sektorze
Artificial Intelligence oraz budowa modelu regresyjnego szacującego roczne zarobki.
Zbiór danych zawiera informacje o ofertach pracy z całego świata, uwzględniając wymogi
techniczne, lokalizację, poziom stanowiska oraz profil pracodawcy.

## Zbiór danych
Dane użyte w projekcie pochodzą z serwisu Kaggle:
[AI and Data Science Job Market Dataset 2020-2026](https://www.kaggle.com/datasets/shree0910/ai-and-data-science-job-market-dataset-20202026)

* **Liczba instancji (rekordów):** 10 345
* **Liczba atrybutów:** 19 (w tym zmienna docelowa)
* **Charakter danych:** syntetyczny — dane zostały wygenerowane algorytmicznie,
  co zostało potwierdzone zarówno przez opis źródłowy na Kaggle, jak i przez
  przeprowadzoną analizę statystyczną

### Atrybuty
1. **job_id** - unikalny identyfikator oferty pracy
2. **job_title** - nazwa stanowiska (np. Data Scientist, AI Engineer)
3. **company_size** - wielkość przedsiębiorstwa (Startup, Medium, MNC itp.)
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
Projekt rozwiązuje **problem regresji**, gdzie zmienną docelową jest przewidywanie
wartości ciągłej `salary` (rocznego wynagrodzenia).

Do trenowania i ewaluacji użyte zostały dwa algorytmy:
1. **Linear Regression (Regresja Liniowa)** — prosty model liniowy służący jako
   punkt odniesienia (baseline).
2. **Random Forest Regressor (Model Lasu Losowego)** — nieliniowy model zespołowy,
   zdolny do uchwycenia złożonych zależności między zmiennymi.

## Wyniki

| Metryka | Random Forest | Regresja Liniowa |
|---------|--------------|-----------------|
| R²      | 0.991        | 0.992           |
| MAE     | 2 543 USD    | 2 447 USD       |
| RMSE    | 2 994 USD    | 2 846 USD       |
| CV R² (5-fold) | 0.990 ± 0.000 | 0.991 ± 0.000 |

Oba modele osiągnęły zbliżone, bardzo wysokie wyniki. Nieznacznie lepsza okazała się
Regresja Liniowa, co jest zaskakujące — zwykle Random Forest radzi sobie lepiej na
danych tabelarycznych. Wyjaśnienie tego zjawiska opisano w sekcji poniżej.

## Analiza cech — dlaczego R² = 0.99?

Tak wysoki wynik dla obu modeli skłonił do głębszej analizy danych. Przeprowadzono
badanie permutation importance oraz rozrzutu średnich wynagrodzeń względem
poszczególnych kategorii:

| Cecha             | Rozrzut średnich wynagrodzeń |
|-------------------|------------------------------|
| experience_level  | 24 597 USD                   |
| job_title         | 20 566 USD                   |
| company_size      |  9 951 USD                   |
| hiring_urgency    |  5 937 USD                   |
| company_industry  |    965 USD                   |
| education_level   |    538 USD                   |
| remote_type       |    185 USD                   |

Analiza wykazała, że zaledwie dwie cechy — `job_title` i `experience_level` —
odpowiadają za dominującą część wariancji wynagrodzenia. Potwierdza to permutation
importance, gdzie obie cechy uzyskały wartości ~0.40 i ~0.25, podczas gdy
`years_experience`, `remote_type` czy `job_posting_year` osiągnęły wartości bliskie 0.

## Eksperyment ablacyjny

Aby potwierdzić powyższe wnioski, wytrenowano dodatkowy model pozbawiony dwóch
dominujących cech (`job_title` i `experience_level`):

| Metryka | Model pełny | Model okrojony | Różnica     |
|---------|-------------|----------------|-------------|
| R²      | 0.991       | 0.177          | **-0.814**  |
| MAE     | 2 545 USD   | 23 652 USD     | +21 107 USD |
| RMSE    | 2 998 USD   | 28 634 USD     | +25 635 USD |

Bez tych dwóch cech R² spada do 0.177, a MAE rośnie niemal 10-krotnie. Oznacza to,
że pozostałe 36 cech łącznie wyjaśniają zaledwie ~18% wariancji wynagrodzeń.

## Wnioski

Wyniki projektu potwierdzają syntetyczny charakter datasetu. Wynagrodzenia zostały
wygenerowane niemal wyłącznie na podstawie `job_title` i `experience_level`,
a pozostałe kolumny (kraj, umiejętności, tryb pracy, wykształcenie) pełnią rolę
dekoracyjną i mają marginalny wpływ na zmienną docelową.

Model osiąga R² = 0.991, jednak nie odzwierciedla to rzeczywistych zależności
rynkowych — model nauczył się wzoru generującego dane, a nie prawdziwych
prawidłowości rynku pracy w branży AI. Na realnych danych wyniki byłyby
znacznie niższe (R² rzędu 0.6–0.8 dla dobrego modelu).

Pipeline projektu jest jednak technicznie poprawny i obejmuje:
- preprocessing bez data leakage (podział train/test przed transformacją)
- kodowanie cykliczne zmiennych czasowych (miesiąc → sin/cos)
- porównanie dwóch algorytmów z cross-walidacją
- analizę ważności cech (permutation importance)
- eksperyment ablacyjny dokumentujący wpływ kluczowych zmiennych