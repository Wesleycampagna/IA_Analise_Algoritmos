import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt

names = ['SepalLength', 'SepalWidth',
         'PetalLength', 'PetalWidth',
         'Class']

df = pd.read_csv('iris.data', names=names)

print("Linhas: %d, Colunas: %d" % (len(df), len(df.columns)))

#Criando features
df['SepalArea'] = df['SepalLength'] * df['SepalWidth']
df['PetalArea'] = df['PetalLength'] * df['PetalWidth']

df['SepalLengthAboveMean'] = df['SepalLength'] > df['SepalLength'].mean()
df['SepalWidthAboveMean'] = df['SepalWidth'] > df['SepalWidth'].mean()

df['PetalLengthAboveMean'] = df['PetalLength'] > df['PetalLength'].mean()
df['PetalWidthAboveMean'] = df['PetalWidth'] > df['PetalWidth'].mean()

# Treinamento - Preparando os dados
features = df.columns.difference(['Class'])

X = df[features].values
y = df['Class'].values

print(features)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

classifier_dt = DecisionTreeClassifier(random_state=1986, criterion='gini', max_depth=3)
classifier_dt.fit(X, y)

#Predicting samples
# Exemplos que serao utilizados para validar o modelo
sample1 = [1.0, 2.0, 3.5, 1.0, 10.0, 3.5, False, False, False, False]  # Iris-setosa
sample2 = [5.0, 3.5, 1.3, 0.2, 17.8, 0.2, False, True, False, False]   # Iris-versicolor
sample3 = [7.9, 5.0, 2.0, 1.8, 19.7, 9.1, True, False, True, True]     # Iris-virginica

print(classifier_dt.predict([sample1, sample2, sample3]))  # Predizendo o tipo da flor

#Cross Validation
from sklearn.model_selection import cross_val_score

scores_dt = cross_val_score(classifier_dt, X, y, scoring='accuracy', cv=5)
print(scores_dt.mean())

#Random Forest
from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(random_state=1986, n_estimators=50, max_depth=5, n_jobs=-1)
classifier_rf.fit(X, y)

scores_rf = cross_val_score(classifier_rf, X, y, scoring='accuracy', cv=5)
print(scores_rf.mean())

#Feature Importance
classifier_rf.fit(X, y)  # Treinando com tudo

features_importance = zip(classifier_rf.feature_importances_, features)
for importance, feature in sorted(features_importance, reverse=True):
    print("%s: %f%%" % (feature, importance*100))

#Grid Search CV
from sklearn.model_selection import GridSearchCV

param_grid = {
            "criterion": ['entropy', 'gini'],
            "n_estimators": [25, 50, 75],
            "bootstrap": [False, True],
            "max_depth": [3, 5, 10],
            "max_features": ['auto', 0.1, 0.2, 0.3]
}
grid_search = GridSearchCV(classifier_rf, param_grid, scoring="accuracy")
grid_search.fit(X, y)

# Pegando o melhor classificador
classifier_rf = grid_search.best_estimator_

print(grid_search.best_score_)
print(grid_search.best_params_)