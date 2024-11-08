import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix as cm
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("../datasets/ucl-finals-juntado.csv", index_col=0)

columnas = ["Año","Nombre","Equipo","Nacionalidad","Edad","Posicion","season",
            "winner","winner-country","score","runner-up","runner-up-country",
            "stadium","final-city","final-country","attendance","winning-way",
            "year","winner_score","runner_up_score"] 

le = LabelEncoder()


for columna in columnas:
    df[columna] = le.fit_transform(df[columna])
#fin for
df_2 = df.drop(columns=['Equipo'])
x = df_2.iloc[:,1:].values
y = df['Equipo']

X_train, X_test, Y_train, Y_test = train_test_split(x,y,train_size=0.80,random_state=0)
arbol = DecisionTreeClassifier()
arbol_equipo = arbol.fit(X_train,Y_train)
fig = plt.figure(figsize=(25,20))
tree.plot_tree(arbol_equipo,
               feature_names=list(df.columns.values),
               class_names=[str(cls) for cls in y.unique()],
               filled=True)
plt.show()
Y_pred = arbol_equipo.predict(X_test)

matriz_confusion = cm(Y_test,Y_pred)
print(matriz_confusion)

precision_global = np.sum(matriz_confusion.diagonal())/np.sum(matriz_confusion)
print(precision_global*100)
