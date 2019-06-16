import load_dataset
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss, make_scorer
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
                ['SomervilleHappinessSurvey2015.txt', ['class', 'a', 'b', 'c', 'd', 'e', 'f']],
                ['breast-cancer.data', ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'class']],
                ['iris.data', ['a', 'b', 'c', 'd', 'class']],
                ['transfusion.data', ['a', 'b', 'c', 'd', 'class']],
                ['balance-scale.data', ['class', 'a', 'b', 'c', 'd']],
                ['breast-cancer-wisconsin.data', ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'class']],
                ['hayes-roth.data', ['a', 'b', 'c', 'd', 'e', 'class']],
                ['mammographic_masses.data', ['a', 'b', 'c', 'd', 'e', 'class']],
                ['australian.dat', ['a1', 'b2', 'c3', 'd4', 'e5', 'f6', 'g7', 'h8', 'i9', 'j10', 'k11', 'l2', 'm13', 'm14', 'class']],
                ['Wholesale customers data.csv', ['class','Region','Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']]]
    
    def __init__(self):

        # carrega os datasets para uso
        print('\ncarregando os dados...')
        self.lds = load_dataset.load_dataset(Main.datasets)
        print('\ncarregamento realizado!')

        # pre processamento 
        print('\npreprocessando os dados...')
        self.lds.prepocess_dataset(onehot=True, normalize=True)  
        print('pre-processamento realizado!')

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


    def get_name_dataset(self):
        return Main.datasets
    
#-----------------------------------------------------------------------------


NUM_ALGRITHM = 6
KNN = 1
DECISION_THEE = 2
NAIVE_BAYES = 3
LOGISTIC_REGRESSION = 4
NEURAL_NETWORK = 5
RUN_GS_RNEURAIS = True   # Como GridSearch demora pakas - False não o faz

trab = Main()


all_means = []
all_vars = []
all_times = []
all_really_means = []
all_really_vars = []
all_really_times = []

def create_folder():
    if not os.path.exists('output-files'):
        os.makedirs('output-files')

def plot_mean_var():
    pass


def get_answers(estimador, x_data, y_data):

    means = []
    variancias = []
    time_to_make_cross_val = []

    for x_fold in estimador:
        
        inicio = time.time()
        placar = cross_val_score(x_fold, X=x_data, y=y_data, cv=trab.get_fold_params())
        fim = time.time()
        
        media = placar.mean()
        variancia = np.std(placar)        

        log_oss = make_scorer(log_loss, needs_proba=True, labels = y_data)
        loss = cross_val_score(x_fold, X=x_data, y=y_data, cv=trab.get_fold_params(), scoring = log_oss)
        media_loss = loss.mean()

        means.append(media)
        variancias.append(variancia)
        time_to_make_cross_val.append(fim - inicio)

        print("tree media eh ", media , " " , "variancia eh ",  variancia, "log loss eh ", media_loss)
    
    all_means.append(means)
    all_vars.append(variancias)
    all_times.append(time_to_make_cross_val)


all_x = trab.get_all_x()
all_y = trab.get_all_y()


# para cada dataset deve-se rodar ao menos 5 combinações de cada algoritmo
for i_dsets in range(len(all_x)):
   
    shape_x = np.array(all_x[i_dsets])
    shape_y = np.array(all_y[i_dsets])

    #print('shape_x: ', shape_x.shape, ' shape_y: ', shape_y.shape)
    # run todos algortmos
    for algorithm in range(NUM_ALGRITHM):

        if algorithm is KNN:

            print('[KNN] - ', trab.get_name_dataset()[i_dsets][0], '\n')

            # salva 4 instancias para teste (parametros diferenciados) -> A 5a vem do GridSearchCV 
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

            # recebe os melhores parametros para o que foi pedido em param_grid
            best_params = trab.get_best_params_grid_search(KNeighborsClassifier(), param_grid, all_x[i_dsets], all_y[i_dsets])

            # executa o estimador conforme os parametros de best_params
            knn_by_gs = KNeighborsClassifier(metric=best_params[0]['metric'], n_neighbors=best_params[0]['n_neighbors'], 
            weights=best_params[0]['weights'])

            # append este 5o elemento para se realizar o cross_validation
            knn.append(knn_by_gs)

            # realiza o cross_validation para cada instancia 
            get_answers(knn, all_x[i_dsets], all_y[i_dsets])
            
            print('-------------------------------------------------------')

        if algorithm is DECISION_THEE:
            
            print('[TREE] - ', trab.get_name_dataset()[i_dsets][0], '\n')

            tree = [ DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=None, 
            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
            max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
            min_impurity_split=None, class_weight=None, presort=False),

            DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=10, 
            min_samples_split=5, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
            max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
            min_impurity_split=None, class_weight=None, presort=False),

            DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=7, 
            min_samples_split=3, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
            max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
            min_impurity_split=None, class_weight=None, presort=True),

            DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, 
            min_samples_split=4, min_samples_leaf=2, min_weight_fraction_leaf=0.0, 
            max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.1, 
            min_impurity_split=None, class_weight=None, presort=False) ]

            param_grid = {'criterion':('gini', 'entropy'), 
            'splitter':('best', 'random'),
            'min_samples_split': (2, 3, 5, 7, 10),
            'max_depth': (1, 2, 3, 5, 7, 10),
            'presort':('auto', True, False)}

            best_params = trab.get_best_params_grid_search(DecisionTreeClassifier(), param_grid, all_x[i_dsets], all_y[i_dsets])

            dtc_by_gs = DecisionTreeClassifier(criterion=best_params[0]['criterion'], splitter=best_params[0]['splitter'], 
            min_samples_split=best_params[0]['min_samples_split'], max_depth=best_params[0]['max_depth'], presort=best_params[0]['presort'])

            tree.append(dtc_by_gs)

            get_answers(tree, all_x[i_dsets], all_y[i_dsets])

            print('-------------------------------------------------------')

        if algorithm is NAIVE_BAYES:

            print('[NBAYES] - ', trab.get_name_dataset()[i_dsets][0], '\n')
                
            gnb = [GaussianNB(priors=None, var_smoothing=1e-09),
            GaussianNB(priors=None, var_smoothing=1e-08),
            GaussianNB(priors=None, var_smoothing=1e-07),
            GaussianNB(priors=None, var_smoothing=1e-10)]

            param_grid = {'var_smoothing':(1e-09, 1e-08, 1e-07)}

            best_params = trab.get_best_params_grid_search(GaussianNB(), param_grid, all_x[i_dsets], all_y[i_dsets])

            gnb_by_gs = GaussianNB(var_smoothing=best_params[0]['var_smoothing'])

            gnb.append(gnb_by_gs)

            get_answers(gnb, all_x[i_dsets], all_y[i_dsets])

            print('-------------------------------------------------------')

        if algorithm is LOGISTIC_REGRESSION:

            print('[LREGRESSION] - ', trab.get_name_dataset()[i_dsets][0], '\n')

            log_regression = [LogisticRegression(penalty='l2', tol=0.0001, class_weight='balanced', solver='lbfgs', max_iter=23000, multi_class='ovr',
            warm_start=False, n_jobs=-1),

            LogisticRegression(penalty='l2', tol=0.001, class_weight='balanced', solver='newton-cg', max_iter=22000, multi_class='ovr',
            warm_start=True, n_jobs=-1),

            LogisticRegression(penalty='l2', tol=0.001, class_weight='balanced', solver='lbfgs', max_iter=31500, multi_class='ovr',
            warm_start=False, n_jobs=-1),

            LogisticRegression(penalty='l2', tol=0.0001, class_weight='balanced', solver='sag', max_iter=17000, multi_class='ovr',
            warm_start=True, n_jobs=-1)]

            param_grid = {'class_weight': ('balanced', None), 'solver':('newton-cg', 'lbfgs'), 
            'max_iter':(45000, 53000), 'warm_start':(False, True), 'multi_class':('ovr', 'auto'), 'n_jobs': (-1, 1)}

            best_params = trab.get_best_params_grid_search(LogisticRegression(), param_grid, all_x[i_dsets], all_y[i_dsets])

            lr_bt_gs = LogisticRegression(class_weight=best_params[0]['class_weight'], solver=best_params[0]['solver'], max_iter=best_params[0]['max_iter'], 
            warm_start=best_params[0]['warm_start'], tol=0.01, multi_class=best_params[0]['multi_class'], n_jobs=best_params[0]['n_jobs'])

            log_regression.append(lr_bt_gs)

            get_answers(log_regression, all_x[i_dsets], all_y[i_dsets])

            print('-------------------------------------------------------')

        if algorithm is NEURAL_NETWORK:

            print('[RNEURAIS] - ', trab.get_name_dataset()[i_dsets][0], '\n')

            r_neurais_clas = [MLPClassifier(hidden_layer_sizes=(10), activation='relu', solver='adam', alpha=0.0005, 
            learning_rate_init=0.01, max_iter=12000, tol=0.0001, n_iter_no_change=10),
            
            MLPClassifier(hidden_layer_sizes=(3), activation='relu', solver='lbfgs', alpha=0.0001, 
            learning_rate_init=0.008, max_iter=8000, tol=0.0001, n_iter_no_change=5),

            MLPClassifier(hidden_layer_sizes=(6), activation='relu', solver='sgd', alpha=0.0004, 
            learning_rate_init=0.005, max_iter=6000, tol=0.0001, n_iter_no_change=15),

            MLPClassifier(hidden_layer_sizes=(50), activation='identity', solver='lbfgs', alpha=0.0008, 
            learning_rate_init=0.001, max_iter=10000, tol=0.0001, n_iter_no_change=3)]

            param_grid = {'hidden_layer_sizes':((5,), (10,)), 'activation':('relu', 'identity'), 
            'alpha':(0.008, 0.0005), 'solver': ('lbfgs', 'sgd'),
            'learning_rate_init':(0.8, 0.5), 'max_iter':(5000, 8000),
            'n_iter_no_change':(10, 15)}

            if RUN_GS_RNEURAIS:
                best_params = trab.get_best_params_grid_search(MLPClassifier(), param_grid, all_x[i_dsets], all_y[i_dsets])

                nn_by_gs = MLPClassifier(hidden_layer_sizes=best_params[0]['hidden_layer_sizes'], activation=best_params[0]['activation'], 
                solver=best_params[0]['solver'], alpha=best_params[0]['alpha'], max_iter=best_params[0]['max_iter'], n_iter_no_change=best_params[0]['n_iter_no_change'])

                r_neurais_clas.append(nn_by_gs)

            else:
                r_neurais_clas.append(MLPClassifier(hidden_layer_sizes=(50), activation='identity', solver='adam', alpha=0.0001, 
                learning_rate_init=0.01, max_iter=14000, tol=0.001, n_iter_no_change=20))

            get_answers(r_neurais_clas, all_x[i_dsets], all_y[i_dsets])

            print('-------------------------------------------------------')


# datasets que tenha colunas muito constratantes (valores) como breast-cancer-wisconsin.data
# knn reglog e rneurais sofrem muita queda da acuracia
# em compensação arvore de decisão apresentou valore  muito bons

# 2 datasets não rodaram na reglog -> breast-cancer-wisconsin.data e Wholesale customers data.csv
# aparentemente por apresentar valores muito altos. 

# reg_log necessita uma enorme quantidade de iteração