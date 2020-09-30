
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

import librosa
import pickle
import math


from sklearn.metrics import r2_score




arq = open('Logs/rede3.txt', 'w')
   
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

print('treino:',len(treino),'teste:',len(teste))



#ajustando X , original formato [mfcc,locid], deixar apenas [mfcc] para treinar o modelo

Aux = []
for i in  range(len(X_ab)):
        Aux.append([])
        Aux[i] = X_ab[i][0]

X_ab= Aux


      

Aux = []
for i in range(len(Y_ab)):
    Aux.append(np.array(Y_ab[i][0]).reshape(-1))

Y_ab = Aux

print('iniciando treino!',file=arq)

r2 = tflearn.metrics.R2()
encoder = tflearn.input_data(shape=[None, 13,int(216)])
encoder = tflearn.dropout(encoder,0.9)
encoder = tflearn.dropout(encoder,0.2)
encoder = tflearn.fully_connected(encoder, 40,activation='crelu')
decoder = tflearn.fully_connected(encoder, int(572), activation='linear')
net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.0007,loss='mean_square', metric=r2)
model = tflearn.DNN(net)
model.fit(X_ab, Y_ab, n_epoch=1000,run_id="auto_encoder", batch_size=128,shuffle=True, show_metric=False)

model.save('Save-Models/Model4-Best.tflearn')


encoding_model = tflearn.DNN(encoder, session=model.session)



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
print("Experimento 11(Locutores conhecidos portugues:",file=arq)
print("acertou: ", acertou,"de: ",tamanho,file=arq) 
print('acuracy:',acertou/tamanho,file=arq)
print('Segmentos -- Treino:',len(X_ab),' Teste:',len(Xtest_ab),file=arq)




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
print("Experimento 5(Locutores não conhecidos portugues:",file=arq)
print("acertou: ", acertou,"de: ",tamanho,file=arq) 
print('acuracy:',acertou/tamanho,file=arq)
print('Segmentos -- Treino:',len(X_ab),' Teste:',len(X_cd),file=arq)


Aux = []
for i in  range(len(X_cd)):
        Aux.append([])
        Aux[i] = X_cd[i][0]

X_cd= Aux


#r2_score calc
Ypredict  = model.predict(X_cd)

r2 = 0

Aux = []
for i in  range(len(Y_cd)):
        Aux.append([])
        Aux[i] = Y_cd[i][0]

Y_cd= Aux


for i in range(len(X_cd)):
    r2  = r2+r2_score(np.array(Y_cd[i]).reshape(-1),np.array(Ypredict[i]).reshape(-1))

r2 = r2/len(X_cd)

print('r2_score: ',r2,file=arq)

### Experimento 6: LibreSpeech

with open('Mfcc-Save/Base3-Cadastrados.txt', 'rb') as f:
        cadastrados_base3 = pickle.load(f)


with open('Mfcc-Save/Base3-Pessoas.txt', 'rb') as f:
        pessoas_base3 = pickle.load(f)

        




X = []

i=0

while i <len(cadastrados_base3):
                            
            X.append([encoding_model.predict([cadastrados_base3[i][0]])[0],cadastrados_base3[i][1]])
            i = i+1          



acertou = 0
tamanho = 0
V = []

for q in range(len(pessoas_base3)):            
                        
        a=[encoding_model.predict([pessoas_base3[q][0]])[0],pessoas_base3[q][1]]

        V.append(a)
                        
        

        
        
        
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
print('Base3(Locutores Librespeech):',file=arq)
print("acertou: ", acertou,"de: ",tamanho,file=arq) 
print('acuracy:',acertou/tamanho,file=arq)
print('Segmentos -- Treino:',len(X_ab),' Teste:',len(pessoas_base3),file=arq)



















