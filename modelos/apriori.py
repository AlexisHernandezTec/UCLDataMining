import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("../datasets/ucl-finals-juntado.csv")  # Cambia el nombre a tu archivo real



df.columns = ['', 'Año', 'Equipo', 'Nombre', 'Nacionalidad', 'Edad', 'Posicion', 
               'season', 'winner-country', 'winner', 'score', 'runner-up', 
              'runner-up-country', 'stadium', 'final-city', 'final-country', 'attendance', 
              'winning-way', 'year', 'winner_score', 'runner_up_score']

print("Nombres ya procesados de columnas:", df.columns)
#df.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'], inplace=True)

if 'Equipo' in df.columns and 'Posicion' in df.columns:
    print("Columnas 'Equipo' y 'Posicion' encontradas en el DataFrame.")
else:
    print("Las columnas 'Equipo' y/o 'Posicion' no se encuentran en el DataFrame después de renombrar.")

if 'Equipo' in df.columns and 'Posicion' in df.columns:
    df['amount'] = df.groupby(['Equipo', 'Posicion'])['Posicion'].transform('count')

df_pivot = df.pivot_table(index='Equipo', columns='Posicion', values='amount', aggfunc='sum').fillna(0)
df_pivot = df_pivot.map(lambda x: x > 0).astype(bool)  # Convertir a tipo booleano

support = 0.01
frequent_items = apriori(df_pivot, min_support=support, use_colnames=True)
frequent_items = frequent_items.sort_values('support', ascending=True)

metric = 'lift'
min_threshold = 3
rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)
rules.reset_index(drop=True).sort_values('confidence', ascending=False, inplace=True)

# Mostrar las reglas
print(rules)
