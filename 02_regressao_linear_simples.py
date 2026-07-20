from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

database=pd.read_csv("Clean_Dataset_Ajustado.csv")

#O atributo mais relevante implementado em uma regressão linear, para predição do atributo alvo determinado
X = database[['class_Business']]
y = database['price']

X_train, X_test, y_train, y_test= train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42)

#treinamento do modelo de regressão linear com o atributo mais relevante
reg = LinearRegression()
reg.fit(X_train, y_train)

#predicao do modelo de regressão linear com o atributo mais relevante
y_pred = reg.predict(X_test)


#Determinação dos valores: RSS, MSE, RMSE e R_squared para esta regressão baseada somente no atributo mais relevante.
rss = np.sum((y_pred - y_test) ** 2)
print("Soma dos quadrados dos erros:",rss)
mse = mean_squared_error(y_test, y_pred)
print("Erro Quadrático Médio:", mse)
rmse = np.sqrt(mse)
print("Raiz do Erro Quadrático médio:", rmse)
r_squared = r2_score(y_test, y_pred)
print("Coeficiente de Determinação:", r_squared)

#Gráfico da reta de regressão em conjunto com a nuvem de atributo.
plt.figure(figsize=(8,6))
plt.scatter(
    X_test,
    y_test,
    color="purple",
    alpha=0.2,
    label="Dados reais"
)
plt.plot(
    X_test.iloc[X_test['class_Business'].argsort()],
    y_pred[X_test['class_Business'].argsort()],
    color="green",
    linewidth=2,
    label="Reta de regressão"
)
plt.title("Regressão Linear")
plt.xlabel("Class Business")
plt.ylabel("Preço")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

