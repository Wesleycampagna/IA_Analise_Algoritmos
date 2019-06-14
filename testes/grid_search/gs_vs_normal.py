from matplotlib import pyplot as plt

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

""" A ideia aqui é de criar dois testes, busca de accuracy pelo método normal de 
DecisionTreeClassifier sem CV apenas com o parametro de random_state e a accuracia 
trazida pelo GridSearchCV com foco em accuracia (linha 38) """

# gera um conjunto de teste randomico
X, y = make_hastie_10_2(n_samples=42, random_state=10)

pd.DataFrame(X)

# separa os dados pra teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y)

#-----------------------------------  DecisionTreeClassifier normal -------------------------

#dtc é o estimador para a arvore de decisão 
dtc1 = DecisionTreeClassifier(min_samples_split=8, random_state=10)
# treino do dataset
dtc1.fit(X_train, y_train)
# predição do dataset 
y_predt1 = dtc1.predict(X_test)

# analise da accuracia obtida
acc1 = accuracy_score(y_test, y_predt1)

#-----------------------------------  DecisionTreeClassifier w/ best param gs? -----------------

#dtc é o estimador para a arvore de decisão 
dtc = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=42,
            min_weight_fraction_leaf=0.0, presort=False, random_state=10,
            splitter='best')
# treino do dataset
dtc.fit(X_train, y_train)
# predição do dataset 
y_predt = dtc.predict(X_test)

# analise da accuracia obtida
acc = accuracy_score(y_test, y_predt)


#-----------------------------------  GridSearch normal -------------------------

# The scorers can be either be one of the predefined metric strings or a scorer
# callable, like the one returned by make_scorer
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

# Setting refit='AUC', refits an estimator on the whole dataset with the
# parameter setting that has the best cross-validated AUC score.
# That estimator is made available at ``gs.best_estimator_`` along with
# parameters like ``gs.best_score_``, ``gs.best_params_`` and
# ``gs.best_index_``
gs = GridSearchCV(DecisionTreeClassifier(random_state=10),
                  param_grid={'min_samples_split': range(2, 403, 10)},
                  scoring=scoring, cv=5, refit='AUC', return_train_score=True, iid=False)

""" Also for multiple metric evaluation, the attributes best_index_, best_score_ and 
best_params_ will only be available if refit is set and all of them will be determined 
w.r.t this specific scorer. best_score_ is not returned if refit is callable. """

gs.fit(X, y)
results = gs.best_score_

#-------------------  A partir dos dados adquiridos, vamos comparar --------------

#print best acuraccy (obs: o best_score aparentemente é média)
print ('best_score - acuracia do gs: {:.3f}'.format(results))
print ('acuracia adaptado com parametros gerados gs: {:.3f}'.format(acc))
print ('acuracia parametros normais: {:.3f}'.format(acc1))

print('--------------------------------------')

print('bonus: (best params? ', gs.best_params_)
print('\nbonus: (best estimator? \n\n\t\t----------------------\n', gs.best_estimator_,
        '\n\t\t----------------------')

#datas = gs.cv_results_

#k = pd.DataFrame(datas)
#print(k.mean()) 


#---------------------------------------------------
#       LEIAM
#---------------------------------------------------

# refit é necesario para poder usar .best_score / .bet_estimator
# porem não sei o que é "AUC"

# Essa parte do doc descreve o modo como pode ser feita o scoring
# que pelo que entendi é o que você quer buscar de objetivo para os 
# parametros de entrada (leia abaixo)

# There are two ways to specify multiple scoring metrics for the scoring parameter:

#As an iterable of string metrics::
#>>> scoring = ['accuracy', 'precision']

#As a dict mapping the scorer name to the scoring function::
#>>> from sklearn.metrics import accuracy_score
#>>> from sklearn.metrics import make_scorer
#>>> scoring = {'accuracy': make_scorer(accuracy_score),
#                 'prec': 'precision'}