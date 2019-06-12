import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score 
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

if __name__ == "__main__":
	X = pd.read_csv("breast-cancer.data", sep=",", header=None)

	le = preprocessing.LabelEncoder()
	le.fit(["no-recurrence-events", "recurrence-events"])
	X[0] = le.transform(X[0])

	le.fit(["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"])
	X[1] = le.transform(X[1])

	le.fit(["lt40", "ge40", "premeno"])
	X[2] = le.transform(X[2])

	le.fit(["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59"])
	X[3] = le.transform(X[3])

	le.fit(["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"])
	X[4] = le.transform(X[4])

	le.fit(["yes", "no", "?"])
	X[5] = le.transform(X[5])


	le.fit(["left", "right"])
	X[7] = le.transform(X[7])


	le.fit(["left_up", 'left_low', "right_up", "right_low", "central", "?"])
	X[8] = le.transform(X[8])

	le.fit(["yes", "no"])
	X[9] = le.transform(X[9])

	# A nossa tabela
	#print(X)

	# Separando valor de classificacao dos atributos
	y = X.pop(7)

	tree = DecisionTreeClassifier()

	# Parametros que queremos testar para o algoritmo escolhido (nesse caso DecisionTree)
	parameters = {'criterion':('gini', 'entropy'), 
				'splitter':('best', 'random'),
				'min_samples_split':(0.25, 0.5, 2, 3),
				'min_impurity_decrease':(0.01, 0.1, 0.5, 1),
				'presort':('auto', True, False)}

	gs = GridSearchCV(tree, parameters, cv = 10, n_jobs = -1, iid = False)

	# Executa GridSearch na database testando todos parametros
	gs.fit(X, y)

	#Todos valores de funcoes que o fit pode retornar
	print(sorted(gs.cv_results_.keys()))

	# Printando todas as possibilidades de acuracias combinando os parametros
	print(gs.cv_results_['mean_test_score'])
