from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys


columnas = ["Año","Nombre","Equipo","Nacionalidad","Edad","Posicion","season",
            "winner-country","score","runner-up","runner-up-country",
            "stadium","final-city","final-country","attendance","winning-way",
            "year","winner_score","runner_up_score"] 

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