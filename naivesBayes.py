import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

df = pd.read_csv('datasets/ucl-finals-juntado.csv')
print(df.head())
le = LabelEncoder()
df['Equipo'] = le.fit_transform(df['Equipo'])


# df.drop(columns=['id'], inplace=True)
# X = df.iloc[:,1:].values
# y = df.iloc[:,0].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=True)
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# classifier = GaussianNB()
# classifier.fit(X_train,y_train)
# y_pred = classifier.predict(X_test)
# cf_matrix = confusion_matrix(y_test,y_pred)
# sns.heatmap(cf_matrix,annot=True, fmt='d', cmap='Blues',cbar=False)

# print(f'Classification report:\n{classification_report(y_test, y_pred)}')
# print(np.concatenate((y_pred.reshape(len(y_pred),1),
#                       y_test.reshape(len(y_test),1)),1))
