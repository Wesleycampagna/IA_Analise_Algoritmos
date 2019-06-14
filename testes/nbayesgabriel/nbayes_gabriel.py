import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score 
#from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

if __name__ == "__main__":

	names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'class']
	X = pd.read_csv("breast-cancer.data", sep=",", header=None, names=names)

	features = X.columns.difference(['class'])

	xx = X[features].values
	yy = X['class'].values

	xx = xx.T

	le = preprocessing.LabelEncoder()
	le.fit(["no-recurrence-events", "recurrence-events"])
	xx[0] = le.transform(xx[0]) 

	le.fit(["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"])
	xx[1] = le.transform(xx[1])

	le.fit(["lt40", "ge40", "premeno"])
	xx[2] = le.transform(xx[2])

	le.fit(["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59"])
	xx[3] = le.transform(xx[3])

	le.fit(["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"])
	xx[4] = le.transform(xx[4])

	le.fit(["yes", "no", "?"])
	xx[5] = le.transform(xx[5])

	le.fit(["left", "right"])
	xx[7] = le.transform(xx[7])

	le.fit(["left_up", 'left_low', "right_up", "right_low", "central", "?"])
	xx[8] = le.transform(xx[8])

	xx = xx.T
	
	text_file = open("output.txt", "w")
	for el1 in range(len(xx)):
		for el in range(len(xx[el1])):
			if el != (len(xx[el1]) -1):
				text_file.write(str(xx[el1][el]) + ', ')
			else: text_file.write(str(xx[el1][el]) + ', ' + yy[el1])
		text_file.write('\n')
	text_file.close()

	gnb = GaussianNB()

	all_accuracies = cross_val_score(estimator=gnb, X=xx, y=yy, cv=10)
	# Acuracias de cada fold
	print(all_accuracies)

	# Media das acuracias
	print(all_accuracies.mean())