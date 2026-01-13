import duckdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


con = duckdb.connect('churn_database.db')

data = pd.DataFrame({
    'user_id': range(1, 1001),
    'monthly_fee': np.random.uniform(20, 150, 1000),
    'tenure_months': np.random.randint(1, 24, 1000),
    'support_calls': np.random.randint(0, 10, 1000),
    'last_login_days_ago': np.random.randint(0, 60, 1000),
    'churn': np.random.choice([0, 1], size=1000, p=[0.8, 0.2])
})

con.execute("CREATE OR REPLACE TABLE customers AS SELECT * FROM data")


query = """
SELECT 
    *,
    (support_calls * 2 + last_login_days_ago) AS risk_score_manual,
    (monthly_fee * tenure_months) AS total_revenue_to_date
FROM customers
"""
df_processed = con.execute(query).df()


X = df_processed[['monthly_fee', 'tenure_months', 'support_calls', 'last_login_days_ago']]
y = df_processed['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'churn_model.pkl')
print("✅ Baza utworzona (churn_database.db)")
print("✅ Model zapisany (churn_model.pkl)")
