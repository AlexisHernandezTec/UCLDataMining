import pandas as pd
df = pd.read_csv("datasets/Jugadores_finales.csv",encoding='latin1') 
df['Nacionalidad'] = df['Nacionalidad'].str.capitalize()
df