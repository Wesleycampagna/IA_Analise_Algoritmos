import load_dataset
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# ------------------------------------------------------------------------------------------
#    Alunos:    William Felipe Tsubota      - 2017.1904.056-7
#               Wesley Souza Campagna       - 2014.1907.010-0
#               Alberto Benites             - 2016.1906.026-4
#               Gabriel Chiba Miyahira      - 2017.1904.005-2
# ------------------------------------------------------------------------------------------

class Main:

    # obrigatorio descrever os names e um ao menos com class
    datasets =  [['heart.dat', ['aa', 'ba', 'ca', 'da', 'ea', 'fa', 'ga', 'ha', 'ia', 'ja', 'ka', 'la', 'ma', 'class']],
                #['Z_Alizadeh_sani_dataset.xlsx', 0], ver a classe dele e fazer os nomes
                ['SomervilleHappinessSurvey2015.txt', ['class', 'a', 'b', 'c', 'd', 'e', 'f']],
                ['../testes/nbayesgabriel/breast-cancer.data', ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'class']]]
    
    def __init__(self):

        # carrega os datasets para uso
        print('\ncarregando os dados...')
        self.lds = load_dataset.load_dataset(Main.datasets)
        print('\ncarregamento realizado!')

        # pre processamento 
        print('\npreprocessando os dados...')
        self.lds.prepocess_dataset()  
        print('pre-processamento realizado!')

        # normalização dos dados
        #self.lds.normalize()      

        # obtem todos os arquivos
        self.datas_x = self.lds.get_datasets_x()
        self.datas_y = self.lds.get_datasets_y()

        print('\n\t\tDATASETS PRONTOS !!\n\n')
        #print(self.datas_y)
        self.cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=False)


    def get_all_x(self):
        return self.lds.get_datasets_x()


    def get_all_y(self):
        return self.lds.get_datasets_y()


    def get_best_params_grid_search(self, estimator, param_grid, x, y):
        grid_search = GridSearchCV(estimator, param_grid, scoring='accuracy', refit=True, cv=self.cross_validation, iid=False)
        grid_search.fit(x, y)
        return grid_search.best_params_, grid_search.best_score_, grid_search.best_estimator_, grid_search.best_index_


    def get_fold_params(self):
        return self.cross_validation


    #def get_folds(self):

    def get_accuracy(y_test, y_predict):
        return accuracy_score(y_test, y_predict)   
    
#-----------------------------------------------------------------------------
NUM_ALGRITHM = 5
KNN = 1
DECISION_THEE = 2
NAIVE_BAYES = 3
LOGISTIC_REGRESSION = 4
NEURAL_NETWORK = 5

trab = Main()

all_x = trab.get_all_x()
all_y = trab.get_all_y()

#print (all_y)

#TODO: Escreve aqui qualquer coia a mais necessaria  - deixei todos os parametros das libs

# para cada dataset deve-se rodar ao menos 5 combinações de cada algoritmo
for i_dsets in range(len(all_x)):

    # run todos algortmos
    for algorithm in range(NUM_ALGRITHM):

        # rodar KNN 5x
        if algorithm is KNN:

            knn = [ KNeighborsClassifier(n_neighbors=5, weights='distance', 
            p=2, metric='chebyshev', metric_params=None),

            KNeighborsClassifier(n_neighbors=9, weights='uniform', metric='euclidean', 
            metric_params=None),

            KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='auto', 
                leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None),
                
            KNeighborsClassifier(n_neighbors=13, weights='distance', algorithm='auto', 
                leaf_size=30, p=2, metric='chebyshev', metric_params=None, n_jobs=None)]
            
            param_grid = {'metric':('euclidean', 'minkowski', 'chebyshev', 'manhattan' ), 
                'n_neighbors':(3,5, 7, 9, 11),
                'weights': ('uniform', 'distance')}

            best_params = trab.get_best_params_grid_search(KNeighborsClassifier(), param_grid, all_x[i_dsets], all_y[i_dsets])

            # consegui rodar com os parametros do gridSearch uhuuuu!!
            knn_by_gs = KNeighborsClassifier(metric=best_params[0]['metric'], n_neighbors=best_params[0]['n_neighbors'], 
            weights=best_params[0]['weights'])

            knn.append(knn_by_gs)

            for kney in knn:
                placar = cross_val_score(kney, X=all_x[i_dsets], y=all_y[i_dsets], cv=trab.get_fold_params())
                media = placar.mean()
                variancia = np.std(placar)
                # jogar isto depois em algo
                print(" media eh ", media , " " , "variancia eh ",  variancia)
            
            print('-------------------------------------------------------')

        # rodar arvore de decisão 5x
        if algorithm is DECISION_THEE:
            
            for i in range(5):

                tree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, 
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                min_impurity_split=None, class_weight=None, presort=False)            
                
                #TODO completar

                param_grid = {'criterion':('gini', 'entropy'), 
                'splitter':('best', 'random'),
                'min_samples_split': np.arange(2, 10),
                'max_depth': np.arange(1,10),
                'presort':('auto', True, False)}

                #print(all_y[i_dsets])
                #print(trab.get_best_params_grid_search(tree, param_grid, all_x[i_dsets], all_y[i_dsets]))


        # rodar naive_bayes 5x
        if algorithm is NAIVE_BAYES:

            for i in range(5):
                
                gnb = GaussianNB(priors=None, var_smoothing=1e-09)
                #TODO completar
            pass

        # rodar regressão logistica 5x
        if algorithm is LOGISTIC_REGRESSION:

            for i in range(5):

                log_regression = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                class_weight=None, random_state=None, solver='warn', max_iter=100, multi_class='warn', verbose=0, warm_start=False, n_jobs=None)
                #TODO completar
            pass

        # rodar redes neurais 5x
        if algorithm is NEURAL_NETWORK:

            for i in range(5):

                r_neurais_clas = MLPClassifier(hidden_layer_sizes=(100), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', 
                learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, 
                verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
                beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

                #TODO completar
            pass
