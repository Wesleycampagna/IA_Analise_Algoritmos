import pandas as pd
import preprocessing as pre_pcs

# ------------------------------------------------------------------------------------------
#    Alunos:    William Felipe Tsubota      - 2017.1904.056-7
#               Wesley Souza Campagna       - 2014.1907.010-0
#               Alberto Benites             - 2016.1906.026-4
#               Gabriel Chiba Miyahira      - 2017.1904.005-2
# ------------------------------------------------------------------------------------------

class load_dataset:

    def __init__(self, array_paths):
        self.datasets = []    
        for entry in array_paths:
            try:
                self.datasets.append(pd.read_csv('../datasets/{}'.format(entry), sep=",", header=None))
                print('SUCCESS \tload dataset {}'.format(entry))
            except:
                try:
                    self.datasets.append(pd.read_excel('../datasets/{}'.format(entry), index_col=None, na_values=['NA']))
                    print('SUCCESS \tload dataset {}'.format(entry))
                except:
                    try:
                        self.datasets.append(pd.read_html('../datasets/{}'.format(entry), index_col=None, na_values=['NA']))
                        print('SUCCESS \tload dataset {}'.format(entry))  
                    except:
                        print('FAIL \t\tload dataset {}'.format(entry))
    

    def prepocess_dataset(self):
        for i in range(len(self.datasets)):
            self.datasets[i] = pre_pcs.label_encoder(self.datasets[i])


    def prepocess_one_dataset(self, i):        
        self.datasets[i-1] = pre_pcs.label_encoder(self.datasets[i-1])


    def get_datasets(self, i=None):
        if i is None:
            return self.datasets

        if i > len(self.datasets):
            return

        return self.datasets[i-1]
