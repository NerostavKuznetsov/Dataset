# Regressão Logistica

from pypandoc import convert_text

content = """# 🧠 Detecção de Fraude com Regressão Logística

Este projeto demonstra como treinar e avaliar um modelo de Machine Learning utilizando Regressão Logística para identificar transações fraudulentas.

## 1. Pré-requisitos

Instale as dependências:

pip install pandas scikit-learn

## 2. Estrutura do Projeto

meu_projeto/
 ├── synthetic_fraud_dataset.csv
 └── modelo_fraude.py

## 3. Script Principal

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("synthetic_fraud_dataset.csv")

X = df[["transaction_amount", "transaction_time"]]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Matriz de confusão:\\n", confusion_matrix(y_test, y_pred))
print("\\nRelatório de classificação:\\n", classification_report(y_test, y_pred))

## 4. Requisitos do Dataset

transaction_amount | transaction_time | is_fraud
123.55             | 15.3             | 0
991.10             | 03.2             | 1

## 5. Como Executar

python modelo_fraude.py

## 6. Saída Esperada

Matriz de Confusão e Relatório de Classificação.

## 7. Melhorias Possíveis

- Novas features
- Normalização
- Validação cruzada
- Random Forest, SVM, XGBoost
- Relatório no formato IEEE
"""

output_path = "/mnt/data/README.txt"
convert_text(content, 'plain', format='md', outputfile=output_path, extra_args=['--standalone'])

output_path
