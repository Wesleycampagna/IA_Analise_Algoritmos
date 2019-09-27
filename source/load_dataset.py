import pandas as pd
import preprocessing as pre_pcs
import copy
import numpy as np

# ------------------------------------------------------------------------------------------
#    Alunos:    William Felipe Tsubota      - 2017.1904.056-7
#               Wesley Souza Campagna       - 2014.1907.010-0
#               Alberto Benites             - 2016.1906.026-4
#               Gabriel Chiba Miyahira      - 2017.1904.005-2
# ------------------------------------------------------------------------------------------

class load_dataset:
    
    def __init__(self, dataset):

        self.datasets = []   
        self.x = [] 
        self.y = []

        self.ARRAY_PATH = 0
        self.NAMES = 1

        for entry in dataset:

            path = ''
            if ('/' or '\\') in entry: path = entry[self.ARRAY_PATH]
            else: path = '../datasets/{}'.format(entry[self.ARRAY_PATH])

            try:
                self.datasets.append(pd.read_csv(path, sep=",", names=entry[self.NAMES]))
                print('SUCCESS \tload dataset {}'.format(path))
            except:
                try:
                    self.datasets.append(pd.read_excel(path, index_col=None, na_values=['NA'], names=entry[self.NAMES]))
                    print('SUCCESS \tload dataset {}'.format(path))
                except:
                    try:
                        self.datasets.append(pd.read_html(path, index_col=None, na_values=['NA'], names=entry[self.NAMES]))
                        print('SUCCESS \tload dataset {}'.format(path))  
                    except:
                        print('FAIL \t\tload dataset {}'.format(path))

        self.split_dataset()
        

    def prepocess_dataset(self, onehot=True, normalize=True):
        for i in range(len(self.datasets)):
            self.x[i], collumns = pre_pcs.label_encoder(self.x[i])
            if normalize:
                self.x[i] = self.normalize(self.x[i])
            if onehot:
                self.x[i] = self.prepocess_one_dataset(self.x[i], collumns)           


    def normalize(self, dataset):
        return pre_pcs.norm(dataset)

    def prepocess_one_dataset(self, dataset, collumns):
        dataset = pre_pcs.one_hot_encoder(dataset, collumns)        
        return dataset


    def split_dataset(self):

        for feat in self.datasets:        
            features = feat.columns.difference(['class'])

            self.x.append(feat[features].values)
            self.y.append(feat['class'].values)

        self.set_porc(self.y)
            

    def set_porc(self, y):

        t = copy.deepcopy(y)
        q = np.array(set(t))
        self.classes = 'x'

        print(q)

        pass
        

    def get_datasets(self, i=None):
        if i is None:
            return self.datasets

        if i > len(self.datasets):
            return

        return self.datasets[i-1]

    
    def get_datasets_x(self):
        return self.x


    def get_datasets_y(self):
        return self.y

    
    def get__():
        return self.classes


datasets =  [['heart.dat', ['aa', 'ba', 'ca', 'da', 'ea', 'fa', 'ga', 'ha', 'ia', 'ja', 'ka', 'la', 'ma', 'class']],
                ['SomervilleHappinessSurvey2015.txt', ['class', 'a', 'b', 'c', 'd', 'e', 'f']]]

ld = load_dataset(datasets)
print(ls.get__())