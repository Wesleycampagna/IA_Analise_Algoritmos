import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

names = ['SepalLength', 'SepalWidth',
         'PetalLength', 'PetalWidth',
         'Class']

dados = pd.read_csv('../../datasets/iris.data', names=names)

print("Linhas: %d, Colunas: %d" % (len(dados), len(dados.columns)))

# Treinamento - Preparando os dados
features = dados.columns.difference(['Class'])

X = dados[features].values
y = dados['Class'].values

print(features)
print(X)

# Para obter as classes como inteiros, utilizamos a classe LabelEncoder da scikit-learn
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
# train_test_split separa o conjunto de dados original
# aleatoriamente em treinamento e teste
# train_size indica a proporcao de objetos presentes
# no conjunto de treinamento (neste caso 70% dos objetos)
# caso deseje-se uma separacao estratificada, deve-se
# informar um parametro adicional stratify=y
X_treino, X_teste, y_treino, y_teste = \
train_test_split(X, y, train_size=0.7, test_size=0.3)
# instancia um knn
# n_neighbors indica a quantidade de vizinhos
# metric indica a medida de distancia utilizada
knn = KNeighborsClassifier(n_neighbors=5,
metric='euclidean')
# treina o knn
knn.fit(X_treino, y_treino)
# testa o knn com X_teste
# y_pred consistira em um numpy.array onde
# cada posicao contem a classe predita pelo knn
# para o respectivo objeto em X_teste
y_pred = knn.predict(X_teste)
print(y_pred)

