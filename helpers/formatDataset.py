from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys


columnas = ['Equipo', 'season', 'winner-country', 'score', 'runner-up', 'runner-up-country',
            'stadium', 'final-city', 'final-country', 'attendance', 'winning-way', 'year',
            'winner_score', 'runner_up_score'] 

le = LabelEncoder()
def formatDSEncoder(df):
    sys.dont_write_bytecode = True
    for columna in columnas:
        df[columna] = le.fit_transform(df[columna])
    #fin for
    df["winner"] = le.fit_transform(df["winner"])
    return df
#end def

def applyLinearRegression(df, predicVar = "winner", testSize = 0.2):
    formatDSEncoder(df)
    y = df[predicVar]
    x = df[columnas]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testSize, random_state=42)

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    sc = regressor.score(x_test, y_test)

    y_pred = regressor.predict(x_test)
    return regressor, y_test, y_pred, sc

#end def

def applyLinearRegression3(df, predicVar="winner", predictColumns=["Equipo"], testSize=0.2):
    """
    Aplica regresión lineal usando las columnas especificadas en predictColumns
    para predecir la columna predicVar.

    Parámetros:
    - df: DataFrame que contiene los datos.
    - predicVar: La columna objetivo que queremos predecir (por defecto "winner").
    - predictColumns: Lista de columnas que se utilizarán como características (por defecto ["Equipo"]).
    - testSize: Proporción del conjunto de prueba (por defecto 0.2).

    Retorna:
    - regressor: Modelo entrenado.
    - y_test: Valores reales del conjunto de prueba.
    - y_pred: Predicciones del conjunto de prueba.
    - sc: Precisión del modelo.
    """
    # Formatear el DataFrame para codificarlo
    df = formatDSEncoder(df)

    # Separar características (X) y variable objetivo (y)
    y = df[predicVar]  # Variable a predecir
    x = df[predictColumns]  # Columnas usadas como entrada

    # Dividir el conjunto en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testSize, random_state=42)

    # Entrenar el modelo de regresión lineal
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Evaluar el modelo
    sc = regressor.score(x_test, y_test)

    # Realizar predicciones
    y_pred = regressor.predict(x_test)

    return regressor, y_test, y_pred, sc
#end def