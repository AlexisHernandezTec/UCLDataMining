import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from sklearn import metrics
from sklearn.metrics import confusion_matrix

#Libreria para visualizar gráficos
# import graficas.model_visuals as mv

df = pd.read_csv("../datasets/ucl-finals-juntado.csv", index_col=0)

columnas = ["Año","Nombre","Equipo","Nacionalidad","Edad","Posicion","season",
            "winner-country","score","runner-up","runner-up-country",
            "stadium","final-city","final-country","attendance","winning-way",
            "year","winner_score","runner_up_score"] 

le = LabelEncoder()
for columna in columnas:
    df[columna] = le.fit_transform(df[columna])
#fin for
df["winner"] = le.fit_transform(df["winner"])


plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Heatmap de Correlaciones")
plt.show()

y = df['winner']
x = df[columnas]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)


sc = regressor.score(x_test, y_test)

print(sc)
# # Predicción sobre el conjunto de prueba
y_pred = regressor.predict(x_test)

# # Gráfico comparativo entre los valores reales y predichos
# plt.grid(True)
# # Obtener los coeficientes del modelo y asociarlos a las columnas

#Llamada dse los metodos de la libreria
# mv.plot_heatmap(df.corr())
# mv.plot_real_vs_predicted(y_test, y_pred)
# mv.plot_pie_chart(regressor.score(x_test, y_test))
# mv.plot_feature_importance(regressor, columnas)
# mv.plot_residual_distribution(y_test, y_pred)
