import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix


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

# Predecir valores
y_pred_continuous = regressor.predict(x_test)

# Convertir valores continuos a clases (umbral: 0.5)
y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred_continuous]

# Verificar clases únicas
print("Clases únicas en y_test:", set(y_test))
print("Clases únicas en y_pred:", set(y_pred))

# Calcular y mostrar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                         display_labels=list(set(y_test)))
display.plot()
plt.title("Matriz de Confusión")
plt.show()
