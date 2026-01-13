import streamlit as st
import pandas as pd
import numpy as np
import joblib
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. KONFIGURACJA I ŁADOWANIE
st.set_page_config(page_title="AI Churn Guard Pro", layout="wide")

try:
    model = joblib.load('churn_model.pkl')
    con = duckdb.connect('churn_database.db')
except Exception as e:
    st.error(f"🚨 Błąd: Brak plików systemowych. Uruchom najpierw 'python train_model.py'.")
    st.stop()

# --- STYLE CSS ---
st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.05); 
        padding: 15px !important; border-radius: 12px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; }
    </style>
    """, unsafe_allow_html=True)

# 2. SIDEBAR - STEROWANIE I IMPORT
st.sidebar.header("⚙️ System sterowania")
threshold = st.sidebar.slider("Próg alarmowy ryzyka (Churn Threshold)", 0.0, 1.0, 0.45)

st.sidebar.divider()
st.sidebar.subheader("📥 Źródło danych")
uploaded_file = st.sidebar.file_uploader("Wgraj plik (CSV/XLSX)", type=['csv', 'xlsx'])

st.sidebar.divider()
st.sidebar.subheader("💰 Revenue Guard")
campaign_cost = st.sidebar.number_input("Koszt kampanii / klient (PLN)", value=25)

# 3. POBIERANIE DANYCH
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
else:
    df = con.execute("SELECT * FROM customers").df()

features = ['monthly_fee', 'tenure_months', 'support_calls', 'last_login_days_ago']

# 4. SILNIK AI I SEGMENTACJA
if all(col in df.columns for col in features):
    # Predykcja Prawdopodobieństwa
    df['churn_probability'] = model.predict_proba(df[features])[:, 1]
    
    # --- SEGMENTACJA KLIENTÓW (K-Means Clustering) ---
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['segment'] = kmeans.fit_predict(scaled_data)
    
    # Mapowanie nazw segmentów na podstawie logiki biznesowej
    df['segment_name'] = df['segment'].map({0: "Srebrny", 1: "Złoty (VIP)", 2: "Brązowy"})
    
    # 5. NAGŁÓWEK I METRYKI GŁÓWNE
    st.title("🛡️ AI Subscription Churn Guard Pro")
    
    at_risk = df[df['churn_probability'] > threshold]
    potential_saved_revenue = at_risk['monthly_fee'].sum()
    total_campaign_cost = len(at_risk) * campaign_cost
    roi = (potential_saved_revenue - total_campaign_cost) / total_campaign_cost if total_campaign_cost > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Zagrożony Przychód", f"{potential_saved_revenue:,.2f} PLN")
    m2.metric("Klienci 'At Risk'", len(at_risk))
    m3.metric("Potencjalny ROI", f"{roi:.1%}")
    m4.metric("Średni Churn", f"{df['churn_probability'].mean():.1%}")

    st.divider()

    # 6. SYSTEM ZAKŁADEK (MODUŁY)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard Analityczny", 
        "🤖 Segmentacja AI", 
        "💸 Revenue Guard (Kalkulator)", 
        "📋 Lista Interwencyjna"
    ])

    with tab1:
        st.subheader("Analiza Czynników i Rozkładu")
        c1, c2 = st.columns(2)
        
        with c1:
            # Wykres Ważności Cech
            importances = model.feature_importances_
            fi_df = pd.DataFrame({'Cecha': ["Cena", "Staż", "Wsparcie", "Brak Logowania"], 'Ważność': importances})
            fig_fi = px.bar(fi_df.sort_values('Ważność'), x='Ważność', y='Cecha', orientation='h', 
                            title="Wpływ cech na decyzję AI", color='Ważność', color_continuous_scale='Reds')
            st.plotly_chart(fig_fi, use_container_width=True)
        
        with c2:
            # Rozkład Churnu
            fig_hist = px.histogram(df, x="churn_probability", nbins=30, title="Rozkład ryzyka w bazie",
                                    color_discrete_sequence=['#636EFA'])
            fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="TWÓJ PRÓG")
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.subheader("Segmentacja Behawioralna AI")
        st.markdown("Algorytm K-Means podzielił Twoich klientów na 3 grupy na podstawie ich zachowań.")
        
        col_s1, col_s2 = st.columns([2, 1])
        
        with col_s1:
            fig_cluster = px.scatter(df, x="tenure_months", y="monthly_fee", color="segment_name",
                                     size="support_calls", hover_data=['user_id'],
                                     title="Mapa Segmentów: Staż vs Opłata",
                                     labels={'tenure_months': 'Staż (mies.)', 'monthly_fee': 'Opłata (PLN)'})
            st.plotly_chart(fig_cluster, use_container_width=True)
        
        with col_s2:
            seg_stats = df.groupby('segment_name')['churn_probability'].mean().reset_index()
            fig_seg_bar = px.bar(seg_stats, x='segment_name', y='churn_probability', 
                                 title="Ryzyko wg Segmentu", color='segment_name')
            st.plotly_chart(fig_seg_bar, use_container_width=True)

    with tab3:
        st.subheader("Analiza Finansowa Ratowania Przychodów")
        
        f1, f2 = st.columns(2)
        with f1:
            # Waterfall Chart
            fig_waterfall = go.Figure(go.Waterfall(
                name = "Przychód", orientation = "v",
                measure = ["relative", "relative", "total"],
                x = ["Obecny Przychód", "Straty (Churn)", "Przychód po Ratowaniu"],
                textposition = "outside",
                text = [f"+{df['monthly_fee'].sum():,.0f}", f"-{potential_saved_revenue:,.0f}", f"{(df['monthly_fee'].sum() - potential_saved_revenue):,.0f}"],
                y = [df['monthly_fee'].sum(), -potential_saved_revenue, 0],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
            ))
            fig_waterfall.update_layout(title = "Wpływ Churnu na Budżet", showlegend = False)
            st.plotly_chart(fig_waterfall, use_container_width=True)
        
        with f2:
            st.write("### Rekomendacja Finansowa")
            if roi > 2:
                st.success(f"**WYSOKA OPŁACALNOŚĆ:** Inwestycja w retencję przyniesie {roi:.1f}x zwrotu.")
            elif roi > 0:
                st.warning("**UMIARKOWANA OPŁACALNOŚĆ:** Rozważ optymalizację kosztów kampanii.")
            else:
                st.error("**BRAK OPŁACALNOŚCI:** Koszt ratowania przewyższa wartość klientów.")

    with tab4:
        st.subheader("Lista priorytetowa dla działu Obsługi Klienta")
        
        # Dodanie akcji rekomendowanej
        def recommend_action(row):
            if row['churn_probability'] > 0.8: return "📞 Telefon VIP + Rabat 30%"
            if row['churn_probability'] > threshold: return "📧 Email z ankietą i bonem"
            return "✅ Monitoruj"
        
        df['Akcja'] = df.apply(recommend_action, axis=1)
        
        view_df = df[df['churn_probability'] > threshold].sort_values('churn_probability', ascending=False)
        st.dataframe(view_df[['user_id', 'segment_name', 'churn_probability', 'Akcja', 'monthly_fee']], use_container_width=True)
        
        csv = view_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Pobierz raport interwencyjny CSV", data=csv, file_name='raport_ai_churn.csv')

else:
    st.error(f"Błąd: Wgrany plik musi posiadać kolumny: {features}")

# 7. SYMULATOR 'WHAT-IF'
st.divider()
with st.expander("🧪 Otwórz Symulator Scenariuszy"):
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1: s_fee = st.number_input("Opłata", 20, 500, 100)
    with sc2: s_tenure = st.slider("Staż (m)", 1, 60, 12)
    with sc3: s_calls = st.slider("Wsparcie", 0, 20, 2)
    with sc4: s_login = st.slider("Dni bez logowania", 0, 60, 5)
    
    res_prob = model.predict_proba(pd.DataFrame([[s_fee, s_tenure, s_calls, s_login]], columns=features))[0][1]
    st.write(f"### Prawdopodobieństwo odejścia dla tego profilu: **{res_prob:.1%}**")