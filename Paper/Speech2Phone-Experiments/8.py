import os
import sys
import numpy as np
seed = 0
import random
random.seed(seed)
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)

import tflearn
import tflearn.metrics
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization


import librosa
import pickle
import math
import scipy



from sklearn.metrics import r2_score



arq = open('Logs/rede9_v2-result-final.txt', 'w')

   
with open('Mfcc-Save/X_a_treino.pickle', 'rb') as f:
        X1 = pickle.load(f)
    

with open('Mfcc-Save/X_b_treino.pickle', 'rb') as f:
        X2 = pickle.load(f)

with open('Mfcc-Save/X_c_treino.pickle', 'rb') as f:
        X4 = pickle.load(f)
    
with open('Mfcc-Save/X_c_teste.pickle', 'rb') as f:
        Xt4 = pickle.load(f)

with open('Mfcc-Save/X_d_teste.pickle', 'rb') as f:
        Xt3 = pickle.load(f)
        
with open('Mfcc-Save/X_d_treino.pickle', 'rb') as f:
        X3 = pickle.load(f)
        
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

treino = X_ab+X3+X4

teste = Xtest_ab+Xt3+Xt4

print('treino:',len(treino),'teste:',len(teste),file=arq)



#ajustando X , original formato [mfcc,locid], deixar apenas [mfcc] para treinar o modelo
Aux = []

for i in  range(len(X_ab)):
        Aux.append([])
        aux = np.asarray(X_ab[i][0])
        aux = scipy.ndimage.zoom(aux, (1.230769231,0.925925926), order=3)
        Aux[i] = aux.reshape(16, 200, 1)
        


X_ab= Aux


      

Aux = []
for i in range(len(Y_ab)):
        aux = np.asarray(Y_ab[i][0]).reshape(-1)
        Aux.append(aux)


Y_ab = Aux

print('iniciando treino!',file=arq)

print(np.array(Y_ab).shape,np.array(X_ab).shape)


encoder = input_data(shape=(None, 16, 200,1))
encoder = tflearn.layers.core.dropout (encoder,0.8)
encoder = conv_2d(encoder, 16, 7, activation='crelu')
print(encoder.get_shape,file=arq)
encoder = max_pool_2d(encoder, [1,5])

# 16x40
print(encoder.get_shape,file=arq)
encoder = conv_2d(encoder, 16, 7, activation='crelu')
encoder = max_pool_2d(encoder, [1,2])
# 16x20
print(encoder.get_shape,file=arq)
encoder = conv_2d(encoder, 8, 7, activation='crelu')
encoder = max_pool_2d(encoder, [2,2]) 
# 8x10
print(encoder.get_shape,file=arq)
encoder =tflearn.layers.normalization.batch_normalization (encoder)

# 4x4
# importante: 5 filtros
print(encoder.get_shape,file=arq)

encoder = fully_connected(encoder, 40, activation='crelu')

decoder = fully_connected(encoder,900, activation='relu')
decoder = tflearn.reshape(decoder, [-1, 1, 900])

decoder = tflearn.layers.recurrent.simple_rnn(decoder, 128,return_seq=True, activation='relu')#,dynamic=True

decoder = tflearn.layers.recurrent.simple_rnn(decoder, 80,return_seq=False, activation='leakyrelu')#,dynamic=True #,dropout=0.5        
# 16x64


decoder = fully_connected(decoder, 572, activation='linear')
network = regression(decoder, optimizer='adam',loss='mean_square', learning_rate=0.0001)


# Training
model = tflearn.DNN(network)
model.fit(X_ab,  Y_ab, n_epoch=1000,shuffle=True, batch_size= 16,show_metric=False, run_id='convnet_embedding')
model.save('Save-Models/Model9.tflearn')


encoding_model = tflearn.DNN(encoder, session=model.session)



X =[]
V = []

#separando primeiro segmento dos locutores para serem comparados.
loc = [0]*21
Aux = []



for i in range(len(Xtest_ab)):
        aux = Xtest_ab[i][0]
        aux = np.asarray(aux)
        aux = scipy.ndimage.zoom(aux, (1.230769231,0.925925926), order=3)
        aux = (np.array(aux.reshape(16, 200, 1)))
            
        if loc[int(Xtest_ab[i][1])] == 0:#segmento do locutor  já cadastrado
                loc[int(Xtest_ab[i][1])] =1
                X.append([np.array(encoding_model.predict([aux])[0]).reshape(-1),Xtest_ab[i][1]])
        else:#segmentos de testes do locutor
                V.append([np.array(encoding_model.predict([aux])[0]).reshape(-1),Xtest_ab[i][1]])
        
        
acertou = 0
tamanho = 0        
posI = 0

for j in range(len(V)):
        
        menordist = math.inf # distancia ficou infinita, assim logo na primeira comparação será atribuido a menor  distancia.
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

print("Experimento 10(Locutores conhecidos portugues:")
print("acertou: ", acertou,"de: ",tamanho) 
print('acuracy:',acertou/tamanho)
print('Segmentos -- Treino:',len(X_ab),' Teste:',len(Xtest_ab))






#r2_score calc


Aux = []
for i in  range(len(Xtest_ab)):
        aux = Xtest_ab[i][0]
        aux = np.asarray(aux)
        aux = scipy.ndimage.zoom(aux, (1.230769231,0.925925926), order=3)
        aux = (np.array(aux.reshape(16, 200, 1)))
        
        Aux.append(aux)
        #Aux[i] = Xtest_ab[i][0]

Xtest_ab= Aux


Aux = []
for i in  range(len(Ytest_ab)):
        aux = np.asarray(Ytest_ab[i][0]).reshape(-1)
        Aux.append(aux)

Ytest_ab= Aux





Ypredict  = model.predict(Xtest_ab)

r2 = 0

for i in range(len(Ytest_ab)):
    r2  = r2+r2_score(np.array(Ytest_ab[i]).reshape(-1),np.array(Ypredict[i]).reshape(-1))

r2 = r2/len(Ytest_ab)

print('r2_score: ',r2,file=arq)




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




X = []
V = []

#separando primeiro segmento dos locutores para serem comparados.
loc = [0]*41
        
for i in range(len(X_cd)):
        aux = X_cd[i][0]
        aux = np.asarray(aux)
        aux = scipy.ndimage.zoom(aux, (1.230769231,0.925925926), order=3)
        aux = (np.array(aux.reshape(16, 200, 1)))
        
        if loc[int(X_cd[i][1])] == 0:#segmento do locutor  já cadastrado
                loc[int(X_cd[i][1])] =1
                X.append([np.array(encoding_model.predict([aux])[0]).reshape(-1),X_cd[i][1]])
        else:#segmentos de testes do locutor
                V.append([np.array(encoding_model.predict([aux])[0]).reshape(-1),X_cd[i][1]])
        
        
        
posI = 0
acertou = 0
tamanho = 0

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




print("Experimento 10 (Locutores não conhecidos portugues:")
print("acertou: ", acertou,"de: ",tamanho) 
print('acuracy:',acertou/tamanho)
print('Segmentos -- Treino:',len(X_ab),' Teste:',len(X_cd))

Aux = []
for i in  range(len(X_cd)):
        aux = X_cd[i][0]
        aux = np.asarray(aux)
        aux = scipy.ndimage.zoom(aux, (1.230769231,0.925925926), order=3)
        aux = (np.array(aux.reshape(16, 200, 1)))
        
        Aux.append(aux)

        
X_cd= Aux


#r2_score calc
Ypredict  = model.predict(X_cd)

r2 = 0

Aux = []

for i in  range(len(Y_cd)):
        aux = np.asarray(Y_cd[i][0]).reshape(-1)
        Aux.append(aux)

Y_cd= Aux


for i in range(len(X_cd)):
    r2  = r2+r2_score(np.array(Y_cd[i]).reshape(-1),np.array(Ypredict[i]).reshape(-1))

r2 = r2/len(X_cd)

print('r2_score: ',r2,file=arq)



















