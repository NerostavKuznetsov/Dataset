import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Exemplo com colunas fictícias
df = pd.read_csv("synthetic_fraud_dataset.csv")

X = df[["transaction_amount"]]   # variável independente
y = df["fraud_score"]            # variável dependente

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Coeficiente:", model.coef_)
print("Intercepto:", model.intercept_)
print("R² no teste:", model.score(X_test, y_test))