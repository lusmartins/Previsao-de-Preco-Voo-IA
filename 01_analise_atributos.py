import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

ds=pd.read_csv("Clean_Dataset.csv")
print("Data Frame inicial:\n",ds)
print("Características do Data Frame:\n",ds.info())

ds.drop(columns=["Unnamed: 0"], inplace=True)  # Removendo a coluna Unnamed: 0, pois não é relevante para a análise do preço
# Verificação de celulas vazias ou Nan
print("\nValores ausentes:")
print(ds.isna().sum())

if ds.isna().sum().sum() > 0:
    ds = ds.dropna()
    print("\nValores ausentes removidos.")
else:
    print("\nNenhum valor ausente encontrado.")


#Conversão das variáveis categóricas em variáveis numéricas
categoricas = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class"
]

ds = pd.get_dummies(
    ds,
    columns=categoricas,
    drop_first=False
)


# Separação do dataset em atributos e alvo (X e y)
X = ds.drop(["price", "flight"], axis= 1)
y = ds["price"].values


# Padronização dos atributos
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)


# modelo de regressão Lasso para determinar a importância dos atributos(quanto cada variável ajuda a explicar o preço)
lasso = Lasso(alpha= 0.1)
lasso.fit(X_scaler, y)


#Atributo mais importante
importancia_atributos = pd.Series(
    lasso.coef_, 
    index= X.columns)
importancia_atributos = importancia_atributos.abs().sort_values(ascending=False)
print("Importância dos atributos:", '\n',importancia_atributos,'\n')
print("Atributo mais importante:", '\n',importancia_atributos.index[0],'\n')


#Gráfico de barras para visualizar a importância dos atributos
plt.figure(figsize=(10,5))
importancia_atributos.plot(kind="bar")
plt.title("Importância dos atributos para previsão do preço")
plt.xlabel("Atributos")
plt.ylabel("Coeficiente Absoluto")
plt.xticks #(rotation=45)
plt.grid(axis="y")
plt.show()


#Dataset com modificações atualizado e salvo
ds.to_csv(
    "Clean_Dataset_Ajustado.csv",
    index=False
)

#esta questão está tentando responder: Quais características do voo mais influenciam o preço da passagem?