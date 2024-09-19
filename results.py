import pandas as pd
df = pd.read_csv('datasets/ucl-finals.csv')
df['year'] = df['season'].str[:2] + df['season'].str[-2:]
df['year'] = df['year'].astype(int)
df = df[df['year'] >= 2010]
df