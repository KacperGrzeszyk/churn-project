# 🛡️ AI Subscription Churn Guard Pro

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green.svg)

Kompleksowy system klasy Enterprise do przewidywania odejść klientów (Churn Prediction) i optymalizacji przychodów (Revenue Operations). Aplikacja łączy zaawansowane uczenie maszynowe z analityką biznesową, pomagając firmom subskrypcyjnym ratować zagrożony przychód.

## 🚀 Kluczowe Funkcje

- **Predykcja Churnu AI**: Wykorzystanie modelu Random Forest do szacowania prawdopodobieństwa odejścia klienta w czasie rzeczywistym.
- **Segmentacja Behawioralna**: Automatyczne grupowanie klientów na segmenty (Złoty VIP, Srebrny, Brązowy) przy użyciu algorytmu **K-Means**.
- **Revenue Guard & ROI**: Moduł finansowy wyliczający zagrożony przychód (MRR) oraz opłacalność planowanych kampanii retencyjnych.
- **Wyjaśnialne AI (XAI)**: Analiza ważności cech (Feature Importance) pokazująca, jakie czynniki (cena, staż, wsparcie) najbardziej wpływają na decyzje modelu.
- **Symulator "What-If"**: Interaktywne narzędzie do testowania scenariuszy i sprawdzania, jak zmiany w zachowaniu klienta wpływają na jego lojalność.
- **Dynamiczny Import/Eksport**: Możliwość wgrywania własnych plików CSV/Excel oraz pobierania gotowych list interwencyjnych dla działu Customer Success.

## 🛠️ Stack Technologiczny

- **Język**: Python
- **Analiza danych**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest, K-Means, StandardScaler)
- **Baza danych**: DuckDB (Fast OLAP)
- **Wizualizacja**: Plotly, Matplotlib
- **Interfejs**: Streamlit

## 📦 Instalacja i Uruchomienie

1. **Sklonuj repozytorium:**
   ```bash
   git clone [https://github.com/TWOJA_NAZWA_UZYTKOWNIKA/churn-guard-ai.git](https://github.com/TWOJA_NAZWA_UZYTKOWNIKA/churn-guard-ai.git)
   cd churn-guard-ai