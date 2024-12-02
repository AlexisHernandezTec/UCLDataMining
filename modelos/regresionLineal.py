import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
#Libreria para visualizar gráficos
import model_visuals as mv

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

# plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label="Predicciones")
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Valor real")
# plt.title("Comparación de Valores Reales vs Predichos")
# plt.xlabel("Valores Reales")
# plt.ylabel("Valores Predichos")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Precisión y error
# score = regressor.score(x_test, y_test)
# error = 1 - score

# # Datos para el gráfico circular
# labels = ['Precisión del Modelo', 'Error']
# sizes = [score, error]
# colors = ['#4CAF50', '#FF5722']  # Verde para precisión, rojo para error
# explode = (0.1, 0)  # Resalta el segmento de precisión

# # Creación del gráfico circular
# plt.figure(figsize=(8, 8))
# plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=explode)
# plt.title("Efectividad del Modelo")
# plt.show()


# # Obtener los coeficientes del modelo y asociarlos a las columnas
# feature_importance = pd.Series(regressor.coef_, index=columnas).sort_values(ascending=False)

# # Crear el gráfico de barras
# plt.figure(figsize=(12, 6))
# feature_importance.plot(kind='bar', color='skyblue', edgecolor='black')
# plt.title("Importancia de las Características en el Modelo de Regresión")
# plt.ylabel("Peso del Coeficiente")
# plt.xlabel("Características")
# plt.xticks(rotation=45, ha='right')  # Rotar etiquetas para mejor visibilidad
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()  # Ajustar diseño para evitar superposiciones
# plt.show()


# # Calcular los residuos
# residuos = y_test - y_pred

# # Gráfico de distribución (histograma y densidad)
# plt.figure(figsize=(10, 6))
# sns.histplot(residuos, kde=True, color='purple', bins=30, alpha=0.7)
# plt.axvline(0, color='red', linestyle='--', label="Punto optimo sin residuos")
# plt.title("Distribución de los Residuos")
# plt.xlabel("Residuos (y_test - y_pred)")
# plt.ylabel("Frecuencia")
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()



#Llamada dse los metodos de la libreria
mv.plot_heatmap(df.corr())
mv.plot_real_vs_predicted(y_test, y_pred)
mv.plot_pie_chart(regressor.score(x_test, y_test))
mv.plot_feature_importance(regressor, columnas)
mv.plot_residual_distribution(y_test, y_pred)






