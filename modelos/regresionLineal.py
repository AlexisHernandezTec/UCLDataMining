import pandas as pd
import matplotlib.pyplot as plt
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
x = df[columnas]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


regressor = LinearRegression()
regressor.fit(x_train, y_train)

sc = regressor.score(x_test, y_test)
print(sc)