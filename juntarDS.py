import pandas as pd

# Cargar datasets
ds1 = pd.read_csv("datasets/Ucl_Naives.csv")
ds2 = pd.read_csv("datasets/Jugadores_finales_limpio.csv", encoding='latin1')

merge_winner = pd.merge(ds2, ds1, left_on=['Equipo', 'Año'], right_on=['winner', 'year'], how='inner')

merge_runner_up = pd.merge(ds2, ds1, left_on=['Equipo', 'Año'], right_on=['runner-up', 'year'], how='inner')

merge_ds = pd.concat([merge_winner, merge_runner_up], ignore_index=True)

merge_ds_sorted = merge_ds.sort_values(by=['Año', 'Equipo'], ascending=[True, True])

merge_ds_sorted.to_csv("datasets/ucl-finals-juntado.csv")
print(merge_ds_sorted)
