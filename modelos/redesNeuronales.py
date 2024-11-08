import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Cargar el archivo CSV
df = pd.read_csv("../datasets/ucl-finals-juntado.csv")

# Asumiendo que "Equipo" es la columna que quieres predecir
X = df[['Año', 'Edad']]  # Cambia esto según las características que necesites
Y = df['Equipo']  # Cambia esto a la columna que deseas predecir

# Codificación de etiquetas para la variable Y
le = LabelEncoder()
Y = le.fit_transform(Y)

# Dividir los datos
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo
mlp = MLPClassifier(hidden_layer_sizes=(32, 16, 32, 16), activation="tanh", max_iter=1000, random_state=42, verbose=True)
mlp.fit(X_train, Y_train)

# Predecir y evaluar
y_pred = mlp.predict(X_test)

# Imprimir resultados
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

class_report = classification_report(Y_test, y_pred)
print("Classification Report: \n", class_report)

# Matriz de confusión
conf_matrix = confusion_matrix(Y_test, y_pred)
print(conf_matrix)  # Verificar la matriz

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, 
                                            display_labels=le.classes_)  # Usar las clases originales
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()