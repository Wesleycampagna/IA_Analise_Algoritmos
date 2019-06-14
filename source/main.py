import load_dataset

# ------------------------------------------------------------------------------------------
#    Alunos:    William Felipe Tsubota      - 2017.1904.056-7
#               Wesley Souza Campagna       - 2014.1907.010-0
#               Alberto Benites             - 2016.1906.026-4
#               Gabriel Chiba Miyahira      - 2017.1904.005-2
# ------------------------------------------------------------------------------------------

class Main:

    datasets =  ['heart.dat',
                'Z_Alizadeh_sani_dataset.xlsx',
                'SomervilleHappinessSurvey2015.txt']
    
    def __init__(self):

        # carrega os datasets para uso
        lds = load_dataset.load_dataset(Main.datasets)

        # pre processamento 
        lds.prepocess_dataset()  

        # normalização dos dados
        #lds.normalize()      

        # obtem todos os arquivos
        self.datas = lds.get_datasets()


    def get_best_params_grid_search(classificador, grid_params):
        pass


    if __name__ == "__main__":
        # codigo dos algs
        # stratifiedKFold
        print('começa aqui')

Main()