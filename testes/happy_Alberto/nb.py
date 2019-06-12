from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# obs: não consegui resolver

dados = pd.read_csv('hap.txt')
print(dados)

aDados = np.array(dados)
print(aDados)

X = aDados[:,:1] # select the first columns
print(X)
Y = aDados[:,1:7] # select de column 1 at 6 (not included column 7)
print(Y)

# randomly splits the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .90) #90% is the train

print("X_train" , X_train)
print("X_test" , X_test)
print("Y_train" , Y_train)
print("Y_test" , Y_test)

#clf = MultinomialNB()

#clf.fit(Y_train, X_train)
#pred = clf.predict([[2, 3, 0, 3, 1]])
#print(pred)


gb = GaussianNB()
# fit the model to the training data
gb.fit(X, Y)

# evaluate with the testing data
print("testing accuracy: ", gb.score(X_train, Y_train) * 100, "%", sep = "")


# modo alternativo
""" #aDados = np.array(dados)

#X = dados[:,:1] # select the first columns
Y = X.pop(0)
#Y = dados[:,1:7] # select de column 1 at 6 (not included column 7)
print(X)
print('----------------------')
print(Y)

# randomly splits the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y) #90% is the train

#clf = MultinomialNB()

#clf.fit(Y_train, X_train)
#pred = clf.predict([[2, 3, 0, 3, 1]])
#print(pred)


gb = GaussianNB()
# fit the model to the training data
gb.fit(X_train, Y_train)

y_predict = gb.predict(X_test)

# evaluate with the testing data
print('testing accuracy: {:.2f}'.format(accuracy_score(Y_test, y_predict)))
 """ 