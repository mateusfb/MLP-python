from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import sys

def read_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path) #lendo o arquivo contendo o dataset
    
    #convertendo o dataframe do pandas em um array do numpy, que 
    #é recebido como parametro pelos classificadores do scikit-learn
    dataset = dataset.to_numpy()

    #Separando os dados e os rótulos de cada instância
    data = dataset[:,:-1]
    labels = dataset[:,-1:].ravel() #a função ravel() transforma o array bidimensional contendo os labels em um array unidimensional

    return data, labels

data, labels = read_dataset('iris.csv') #lendo o dataset

'''
construindo o classificador, onde:
-hidden_layer_sizes: tupla onde o i-ésimo elemento representa o número de nêurons na i-ésima hidden layer 
                        (o tamanho da tupla define o número de camadas)
-activation: função de ativação da hidden layer
                (pode ser uma das funções do conjunto {‘identity’, ‘logistic’, ‘tanh’, ‘relu’})
-solver: algoritmo de otimização
            (pode ser uma das funções do conjunto {‘lbfgs’, ‘sgd’, ‘adam’})
-max_iter: número máximo de iterações
-random_state: seed para geração randômica de pesos e bias
'''
mlp = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',
                        random_state=1)

mlp.fit(data, labels) #Treinando o classificador com o dados e os rótulos do dataset

to_be_classified = [[6.3,3.3,6.0,2.5]] #definindo as instâncias que serão classidicados, neste caso apenas uma
predicted = mlp.predict(to_be_classified) #predizendo o rótulo das instâncias presentes em to_be_classified e armazenando em predited

print(predicted)