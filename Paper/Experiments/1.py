import speaker.recognition as SR
import os
import time
from python_speech_features import mfcc #instalacao via pip: pip install python_speech_features
import scipy.io.wavfile as wav
import librosa

model = "modelo-gmm.modelo"
train_data = 'bases/base-gmm/Treino/'
test_data = 'bases/base-gmm/Teste/'
GmM = SR.GMMRec() # Create a new recognizer
X = []
    
#os.system("rm *.txt")
audio_files = os.listdir(train_data )
i=0

while i <len(audio_files):
        
            #a = data.one_hot_from_item(data.speaker(f), speakers)
            #print("return:",a)
            f = audio_files[i]
            if (f[-4::] == ".wav"):
                nome,a= f.split('.')
                print(nome)
                #print(test_data + a + ".wav")
                #print(train_data + f)
           
                (sr, sig) = wav.read(train_data + f)
                novo_mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=13)
    
                GmM.enroll(nome, novo_mfcc)
                print("enroll")
    
                
                                
                
                                
                                
                #X.append([encoding_model.predict([librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)])[0],nome])
                                
                                
                    

            i = i+1

print("treinando")           
GmM.train() # Treinar modelo GMM
print("treinado")
#GmM.dump(model) # salvar modelo para verificacao posterior de vozes

         
#results = reconhecerr.predict(mfcc_vecs)[0]


acertou = 0


audio_files = os.listdir(test_data )
tamanho = len(audio_files)
i=0
while i <len(audio_files):
    f = audio_files[i]
    nome,a= f.split('-')
    (sr, sig) = wav.read(test_data+f)
    a = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=13) 
    b = GmM.predict(a)[0]
    print(b , nome)
    if(str(b) == str(nome)):
                             
        acertou = acertou +1
    i=i+1





                             

    
    
    


     


                
        

    
print("acertou: ", acertou,"de: ",tamanho) 
print(acertou/tamanho)
