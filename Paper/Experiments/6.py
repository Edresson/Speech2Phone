import random
random.seed(0)
import numpy as np
np.random.seed(0)

import tflearn
import numpy as np
import sys
import pickle


from tensorflow import reset_default_graph
reset_default_graph()

from scipy import spatial

import math
import itertools
arq = open('Logs/exp2.4-test.txt', 'w')


### xval = novos locutores ...
      
net = tflearn.input_data(shape=[None, 80*2])
net = tflearn.dropout(net, 0.9)
net = tflearn.fully_connected(net,80,activation='relu',regularizer='L2')


net = tflearn.fully_connected(net,8,activation='crelu',regularizer='L2')

net = tflearn.fully_connected(net,2 , activation='softmax')
acc = tflearn.Accuracy()
network = tflearn.regression(net, optimizer='adam',metric = acc ,loss='categorical_crossentropy',learning_rate=0.00005 )
#criando a rede .
model = tflearn.DNN(network, tensorboard_verbose=0,tensorboard_dir='tflearn_logs')


model.load('./Save-Models/Model5-rede-dissernente.tflearn')


acerto = 0



with open('Mfcc-Save/Embedding-rede-discernente-test.pickle', 'rb') as f:
        X_d = pickle.load(f)
    





X =[]
V = []

#separando primeiro segmento dos locutores para serem comparados.
loc = [0]*41
        
for i in range(len(X_d)):
        if loc[int(X_d[i][1])] == 0:#segmento do locutor  já cadastrado
                loc[int(X_d[i][1])] =1
                X.append(X_d[i])
        else:#segmentos de testes do locutor
                V.append(X_d[i])
        
        

#com o não
posI = 0
acertou = 0
tamanho = 0
# sim = [1,0], não é [0,1]
for j in range(len(V)):
        
        menordist = math.inf
        i =0
            
        while i < len(X):
                
                #distancia = np.sqrt(sum([(xi-yi)**2 for xi,yi in zip(V[j][0],X[i][0])]))
                
                distancia =(model.predict([list(itertools.chain(V[j][0],X[i][0]))])[0])[0] #1 pois queremos minimizar a change de não ser o locutor 
                
                
                
                print(distancia,V[j][1],X[posI][1])      
                if distancia <  menordist:
                    menordist = distancia
                    posI= i
                    
                i=i+1
        
        
        if(X[posI][1] == V[j][1]):
                acertou = acertou +1
                

       


tamanho = len(V)    
print("Experimento 6(Locutores não conhecidos portugues:",file=arq)
print("acertou: ", acertou,"de: ",tamanho,file=arq) 
print('acuracy:',acertou/tamanho,file=arq)
print('Segmentos -- Treino:',' Teste:',len(X_d),file=arq)


#com o sim

posI = 0
acertou = 0
tamanho = 0
# sim = [1,0], não é [0,1]
for j in range(len(V)):
        
        menordist = 0
        i =0
            
        while i < len(X):
                
                #distancia = np.sqrt(sum([(xi-yi)**2 for xi,yi in zip(V[j][0],X[i][0])]))
                
                distancia =(model.predict([list(itertools.chain(V[j][0],X[i][0]))])[0])[0] #0 pois queremos maximizar a probabilidade de ser o locutor 
                
                
                
                print(distancia,V[j][1],X[posI][1])      
                if distancia >  menordist:
                    menordist = distancia
                    posI= i
                    
                    
                i=i+1
        
        
        if(X[posI][1] == V[j][1]):
                acertou = acertou +1
                

       


tamanho = len(V)    
print("Experimento 6 (Locutores não conhecidos portugues:",file=arq)
print("acertou: ", acertou,"de: ",tamanho,file=arq) 
print('acuracy:',acertou/tamanho,file=arq)
print('Segmentos -- Treino:',' Teste:',len(X_d),file=arq)



        
