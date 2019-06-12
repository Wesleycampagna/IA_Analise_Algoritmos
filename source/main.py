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

        # por hora teste de instancias
        lds.prepocess_dataset()        
        print(len(lds.get_datasets()))
        print(len(lds.get_datasets(1)))


Main()