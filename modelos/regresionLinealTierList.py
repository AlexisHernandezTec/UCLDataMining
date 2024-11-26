import pandas as pd
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Cargar datos
df = pd.read_csv("../datasets/ucl-finals-juntado.csv", index_col=0)

columnas = ["Año", "Nombre", "Equipo", "Nacionalidad", "Edad", "Posicion", "season",
            "winner-country", "score", "runner-up", "runner-up-country",
            "stadium", "final-city", "final-country", "attendance", "winning-way",
            "year", "winner_score", "runner_up_score"]

le = LabelEncoder()
for columna in columnas:
    df[columna] = le.fit_transform(df[columna])
df["winner"] = le.fit_transform(df["winner"])

y = df['winner']

def top_and_bottom_combinations(df, target, columns, top_n=5):
    results = []
    
    # Probar todas las combinaciones posibles
    for r in range(1, len(columns) + 1):
        for combo in itertools.combinations(columns, r):
            # Crear subconjunto de datos con las columnas de la combinación actual
            x = df[list(combo)]
            x_train, x_test, y_train, y_test = train_test_split(x, target, test_size=0.1, random_state=0)
            
            # Entrenar el modelo y calcular el score
            regressor = LinearRegression()
            regressor.fit(x_train, y_train)
            score = regressor.score(x_test, y_test)
            
            # Guardar la combinación y su score
            results.append((combo, score))
    
    # Ordenar por score en orden descendente
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Obtener las 5 mejores y 5 peores combinaciones
    top_results = results[:top_n]
    bottom_results = results[-top_n:]
    
    return top_results, bottom_results

# Ejecutar la función
top_5, bottom_5 = top_and_bottom_combinations(df, y, columnas)

print("Top 5 combinaciones:")
for combo, score in top_5:
    print(f"Combinación: {combo}, Score: {score}")

print("\nBottom 5 combinaciones:")
for combo, score in bottom_5:
    print(f"Combinación: {combo}, Score: {score}")
