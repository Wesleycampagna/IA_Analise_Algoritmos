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
    
    #names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'class']
    #dataset = pd.read_csv("../testes/nbayesgabriel/breast-cancer.data", sep=",", header=None, names=names)    
#    dataset = [['vhigh','vhigh',2,'big','low','unacc'],
#                ['vlow','vlow',2,'small','med','unacs'],
#                ['vmedium','vmed',2,'medii','haigh','unacscc']]
    
    #print(dataset)
    
    #print("--------------------------")
    
#    print(dataset)
#    print("--------------------------")
    
    dataset = np.array(dataset)

    
    #print(dataset)
    #print("--------------------------")
    
    #dataset = [*zip(*dataset)]
    dataset = dataset.T
    #print(dataset[6])
    #print("--------------------------")
    
    
    collumns = []
    strings = []
    linhas = len(dataset)
    colunas = len(dataset[0])
    le = preprocessing.LabelEncoder()
    for i in range (0, linhas):
        for j in range(0, colunas):
            #print(dataset[j][i])
            if (isinstance(dataset[i][j], str)):
                if(i not in collumns):
                    collumns.append(i)
                if(dataset[i][j] not in strings):
                    strings.append(dataset[i][j])
        #print(strings)
        #print("-----------------------")
        if(i in collumns):
            #print((dataset[i]))
            #print(strings)
            le.fit(strings)
            (dataset[i]) = le.transform((dataset[i]))
        strings = []
        
        
    #print(dataset)
    #print("-----------------------")
        

    #dataset = [*zip(*dataset)]
    dataset = dataset.T
    
    #dataset = [list(row) for row in dataset]
    dataset = dataset
        
#    print(dataset)
#    print("-----------------------")
#    print(collumns)
                    
        
            
            

    return dataset, collumns


def one_hot_encoder(dataset, collumns):
    return dataset


##label_encoder()