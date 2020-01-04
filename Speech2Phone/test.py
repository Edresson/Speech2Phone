

import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
tf.set_random_seed(0)
import tflearn
import numpy as np
import sys
import pickle
    
from scipy import spatial

import librosa
import pickle
import math

arq = open('debug.txt','w')


from sklearn.metrics import r2_score

#Load Model3
encoder = tflearn.input_data(shape=[None, 13,int(216)])
encoder = tflearn.dropout(encoder,0.9)
encoder = tflearn.dropout(encoder,0.2)
encoder = tflearn.fully_connected(encoder, 40,activation='crelu')
decoder = tflearn.fully_connected(encoder, int(572), activation='linear')
net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.0007,loss='mean_square', metric=None)#categorical_crossentropy

model = tflearn.DNN(net, tensorboard_verbose=0,tensorboard_dir='tflearn_logs')


model.load('./Saver-Model/Model4-Best-40loc.tflearn')

encoding_model = tflearn.DNN(encoder, session=model.session)


#Exp3:
with open('Mfcc-Save/X_a_treino.pickle', 'rb') as f:
        X1 = pickle.load(f)
    

with open('Mfcc-Save/X_b_treino.pickle', 'rb') as f:
        X2 = pickle.load(f)
       
with open('Mfcc-Save/X_a_teste.pickle', 'rb') as f:
        Xt1 = pickle.load(f)
          
with open('Mfcc-Save/X_b_teste.pickle', 'rb') as f:
        Xt2 = pickle.load(f)

with open('Mfcc-Save/Y_a_treino.pickle', 'rb') as f:
        Y1 = pickle.load(f)
    

with open('Mfcc-Save/Y_b_treino.pickle', 'rb') as f:
        Y2 = pickle.load(f)
       
with open('Mfcc-Save/Y_a_teste.pickle', 'rb') as f:
        Yt1 = pickle.load(f)
          
with open('Mfcc-Save/Y_b_teste.pickle', 'rb') as f:
        Yt2 = pickle.load(f)
        

          
X_ab = X1+X2
Xtest_ab = Xt1+Xt2
    
Y_ab= Y1+Y2
Ytest_ab = Yt1+Yt2

X =[]
V = []

#separando primeiro segmento dos locutores para serem comparados.
loc = [0]*21
        
for i in range(len(Xtest_ab)):
        if loc[int(Xtest_ab[i][1])] == 0:#segmento do locutor  já cadastrado
                loc[int(Xtest_ab[i][1])] =1
                X.append([encoding_model.predict([Xtest_ab[i][0]])[0],Xtest_ab[i][1]])
        else:#segmentos de testes do locutor
                V.append([encoding_model.predict([Xtest_ab[i][0]])[0],Xtest_ab[i][1]])
        
        
        
acertou = 0
tamanho = 0        
        
posI = 0

for j in range(len(V)):
        
        menordist = math.inf
        i =0
            
        while i < len(X):
                
                distancia = np.sqrt(sum([(xi-yi)**2 for xi,yi in zip(V[j][0],X[i][0])]))
                
                if distancia <  menordist:
                    menordist = distancia
                    posI= i
                i=i+1
        
        
        if(X[posI][1] == V[j][1]):
                acertou = acertou +1
                

       


tamanho = len(V)      
print("Experimento 3(Locutores conhecidos portugues:")
print("acertou: ", acertou,"de: ",tamanho) 
print('acuracy:',acertou/tamanho)
print('Segmentos -- Treino:',len(X_ab),' Teste:',len(Xtest_ab),'/n')




#r2_score calc


Aux = []
for i in  range(len(Xtest_ab)):
        Aux.append([])
        Aux[i] = Xtest_ab[i][0]

Xtest_ab= Aux


Aux = []
for i in  range(len(Ytest_ab)):
        Aux.append([])
        Aux[i] = Ytest_ab[i][0]

Ytest_ab= Aux


Ypredict  = model.predict(Xtest_ab)

r2 = 0

for i in range(len(Ytest_ab)):
    r2  = r2+r2_score(np.array(Ytest_ab[i]).reshape(-1),np.array(Ypredict[i]).reshape(-1))

r2 = r2/len(Ytest_ab)

print('r2_score: ',r2)




#Experimento 4

with open('Mfcc-Save/X_c_treino.pickle', 'rb') as f:
        Xc = pickle.load(f)
    

with open('Mfcc-Save/X_d_treino.pickle', 'rb') as f:
        Xd = pickle.load(f)
       
with open('Mfcc-Save/X_c_teste.pickle', 'rb') as f:
        Xtc = pickle.load(f)
          
with open('Mfcc-Save/X_d_teste.pickle', 'rb') as f:
        Xtd = pickle.load(f)

with open('Mfcc-Save/Y_c_treino.pickle', 'rb') as f:
        Yc = pickle.load(f)
    

with open('Mfcc-Save/Y_d_treino.pickle', 'rb') as f:
        Yd = pickle.load(f)
       
with open('Mfcc-Save/Y_c_teste.pickle', 'rb') as f:
        Ytc = pickle.load(f)
          
with open('Mfcc-Save/Y_d_teste.pickle', 'rb') as f:
        Ytd = pickle.load(f)


X_cd = Xc+Xd
Xtest_cd = Xtc+Xtd
    
Y_cd= Yc+Yd
Ytest_cd = Ytc+Ytd

X_cd=X_cd+Xtest_cd
Y_cd = Y_cd+Ytest_cd




X =[]

V = []

#separando primeiro segmento dos locutores para serem comparados.
loc = [0]*41
        
for i in range(len(X_cd)):
        if loc[int(X_cd[i][1])] == 0:#segmento do locutor  já cadastrado
                loc[int(X_cd[i][1])] =1
                X.append([encoding_model.predict([X_cd[i][0]])[0],X_cd[i][1]])
        else:#segmentos de testes do locutor
                V.append([encoding_model.predict([X_cd[i][0]])[0],X_cd[i][1]])
        
acertou = 0
tamanho = 0       
posI = 0

for j in range(len(V)):
        
        menordist = math.inf
        i =0
            
        while i < len(X):
                
                distancia = np.sqrt(sum([(xi-yi)**2 for xi,yi in zip(V[j][0],X[i][0])]))
                
                if distancia <  menordist:
                    menordist = distancia
                    posI= i
                i=i+1
        
        print(len(X),len(V),posI, j, X[posI][1], V[j][1], X[posI][0][0:3], V[j][0][0:3],file=arq)
        if(X[posI][1] == V[j][1]):
                acertou = acertou +1
                

       


tamanho = len(V)    
print("Experimento 4(Locutores não conhecidos portugues:")
print("acertou: ", acertou,"de: ",tamanho) 
print('acuracy:',acertou/tamanho)
print('Segmentos -- Treino:',len(X_ab),' Teste:',len(X_cd),'/n')


Aux = []
for i in  range(len(X_cd)):
        Aux.append([])
        Aux[i] = X_cd[i][0]

X_cd= Aux


#r2_score calc
Ypredict  = model.predict(X_cd[:100])

r2 = 0

Aux = []
for i in  range(len(Y_cd[:100])):
        Aux.append([])
        Aux[i] = Y_cd[i][0]

Y_cd= Aux


for i in range(len(X_cd[:100])):
    r2  = r2+r2_score(np.array(Y_cd[i]).reshape(-1),np.array(Ypredict[i]).reshape(-1))

r2 = r2/len(X_cd[:100])

print('r2_score: ',r2)


