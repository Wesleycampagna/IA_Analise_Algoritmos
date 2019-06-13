from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

names = ['class', 'servicos',
         'moradia', 'escola',
         'policia', 'ruas', 'comunidade']

df = pd.read_csv('hapiness.txt', names=names)

print("Linhas: %d, Colunas: %d" % (len(df), len(df.columns)))

# Treinamento - Preparando os dados
features = df.columns.difference(['class'])

X = df[features].values
y = df['class'].values

print(features)

gb = GaussianNB()
# fit the model to the whole data
gb.fit(X, np.ravel(y))

# evaluate with the testing whole data
print("Testing accuracy: ", gb.score(X, y) * 100, "%", sep = "")

# randomly splits the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, np.ravel(y), test_size = .9) #90%  - test and 10% - train
print(X_test)

# fit the model to the training data
gb.fit(X_train, Y_train)

# evaluate with the testing data
print("Testing accuracy with split test=90%: ", gb.score(X_train, Y_train) * 100, "%", sep = "")

print(gb.predict(X_test))