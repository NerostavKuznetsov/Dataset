import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Exemplo com colunas fictícias
df = pd.read_csv("synthetic_fraud_dataset.csv")

X = df[["transaction_amount", "transaction_time"]]  # variáveis independentes
y = df["is_fraud"]                                  # variável dependente binária

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))