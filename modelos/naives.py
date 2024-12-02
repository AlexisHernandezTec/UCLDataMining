
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

le = LabelEncoder()
df = pd.read_csv("../datasets/ucl-finals-juntado.csv", index_col=0)

columnas = ["Año","Nombre","Equipo","Nacionalidad","Edad","Posicion","season",
            "winner","winner-country","score","runner-up","runner-up-country",
            "stadium","final-city","final-country","attendance","winning-way",
            "year","winner_score","runner_up_score"] 

for columna in columnas:
    df[columna] = le.fit_transform(df[columna])
#fin for
df_2 = df.drop(columns=['Equipo'])
x = df_2.iloc[:,1:].values
y = df['Equipo']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=True)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
cf_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix,annot=True, fmt='d', cmap='Blues',cbar=False)

print(f'Classification report:\n{classification_report(y_test, y_pred)}')
print(np.concatenate((y_pred.reshape(len(y_pred),1),
                      y_test.reshape(len(y_test),1)),1))