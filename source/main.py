import load_dataset

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
                ['SomervilleHappinessSurvey2015.txt', ['class', 'a', 'b', 'c', 'd', 'e']],
                ['../testes/nbayesgabriel/breast-cancer.data', ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'class']]]
    
    def __init__(self):

        # carrega os datasets para uso
        lds = load_dataset.load_dataset(Main.datasets)

        # pre processamento 
        lds.prepocess_dataset()  

        # normalização dos dados
        #lds.normalize()      

        # obtem todos os arquivos
        self.datas_x = lds.get_datasets_x()
        self.datas_y = lds.get_datasets_y()

        #print(self.datas_y)
        
    def get_best_params_grid_search(classificador, grid_params):
        pass

Main()

#TODO: Escreve aqui 