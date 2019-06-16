# ------------------------------------------------------------------------------------------
#    Alunos:    William Felipe Tsubota      - 2017.1904.056-7
#               Wesley Souza Campagna       - 2014.1907.010-0
#               Alberto Benites             - 2016.1906.026-4
#               Gabriel Chiba Miyahira      - 2017.1904.005-2
# ------------------------------------------------------------------------------------------

from sklearn import preprocessing
import numpy as np
import pandas as pd


def label_encoder(dataset):

    dataset = np.array(dataset)
    dataset = dataset.T

    collumns = []
    strings = []
    linhas = len(dataset)
    colunas = len(dataset[0])

    le = preprocessing.LabelEncoder()

    for i in range (0, linhas):
        for j in range(0, colunas):
            if (isinstance(dataset[i][j], str)):
                if(i not in collumns):
                    collumns.append(i)
                if(dataset[i][j] not in strings):
                    strings.append(dataset[i][j])
                    
        if(i in collumns):
            le.fit(strings)
            (dataset[i]) = le.transform((dataset[i]))

        strings = []
        
    dataset = dataset.T
    
    return dataset, collumns


def one_hot_encoder(dataset, collumns):

    one_hot = preprocessing.OneHotEncoder(sparse=True, categories='auto')
    dilatation = 0

    for i in range(len(dataset)):

        if (i in collumns):

            # para extrair a coluna precisou-se do numpy
            dataset = np.array(dataset)
            # cria-se um novo vetor com 1 colula - posição zero
            collumn_to_transform = dataset[:, dilatation]
            # reshape do vetor
            collumn_to_transform.shape = (len(dataset), 1)

            # convert the collumn on binary with oneHotEncoder
            one_hot.fit(collumn_to_transform)
            new_values = one_hot.transform(collumn_to_transform).toarray()

            # return dataset to list to insert new elements
            dataset = dataset.tolist()

            # add new elements
            for i in range(len(dataset)):
                dataset[i][dilatation:dilatation+1] = new_values[i]

            # tentado::dataset = np.insert(dataset, [collumn + dilatation], new_values, axis=1)
            dilatation += (len(new_values[0]))

        else:
            #condição de parada e ajuste
            if dilatation < len(dataset):
                dilatation += 1
            else: break;

    return np.array(dataset)

##label_encoder()


def norm(dataset):
    feature_scaler = preprocessing.MinMaxScaler()
    return feature_scaler.fit_transform(dataset)