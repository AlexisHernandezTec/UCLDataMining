import pandas as pd
from deep_translator import GoogleTranslator
df = pd.read_csv('datasets/ucl-finals.csv')
df['year'] = df['season'].str[:2] + df['season'].str[-2:]
df['year'] = df['year'].astype(int)
df = df[df['year'] >= 2010]

def translate(text):
    translator = GoogleTranslator(source='en', target='es')
    return translator.translate(text)
#end def
df['winner-country'] = df['winner-country'].apply(translate).str.capitalize().replace('Pavo', 'Turquía')
df['runner-up-country'] = df['runner-up-country'].apply(translate).str.capitalize().replace('Pavo', 'Turquía')
df['final-country'] = df['final-country'].apply(translate).str.capitalize().replace('Pavo', 'Turquía')
df['final-city'] = df['final-city'].apply(translate).str.capitalize().replace('Pavo', 'Turquía')
df

# runner-up-country	
# final-country
# final-city 