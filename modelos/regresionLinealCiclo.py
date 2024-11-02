import pandas as pd
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder



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
       
y = df['winner']



def best_combination(df, target, columns):
    best_score = -1
    best_combo = []
    
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
            print(combo, score)
            
            # Actualizar si el score es mejor
            if score > best_score:
                best_score = score
                best_combo = combo
    
    return best_combo, best_score

# Ejecutar la función
mejor_combinacion, mejor_score = best_combination(df, y, columnas)

print("Mejor combinación de columnas:", mejor_combinacion)
print("Score más alto:", mejor_score)
    