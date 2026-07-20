from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd

database=pd.read_csv("Clean_Dataset_Ajustado.csv")

#processo de  regressão e realização de um gridsearch cross-validation

atributo_importante = "class_Business"

X = database[[atributo_importante]]
y = database['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Melhor parametrização para os regressores de Lasso e Ridge.
parameters = {'alpha': [0.001,0.01, 0.1, 1, 10,100]}

lasso = Lasso()
lasso_grid = GridSearchCV(
    lasso, 
    parameters, 
    cv=5)
lasso_grid.fit(X_train, y_train)


ridge = Ridge()
ridge_grid = GridSearchCV(
    ridge, 
    parameters, 
    cv=5)
ridge_grid.fit(X_train, y_train)

#Print das melhores configurações de cada scores
print("Melhores configurações para Lasso:")
print(lasso_grid.best_params_)
print("Melhor score para Lasso:")
print(lasso_grid.best_score_)

print("Melhores configurações para Ridge:")
print(ridge_grid.best_params_)
print("Melhor score para Ridge:")
print(ridge_grid.best_score_)