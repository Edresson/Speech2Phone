import speaker.recognition as SR
import os
import time
from python_speech_features import mfcc #instalacao via pip: pip install python_speech_features
import scipy.io.wavfile as wav


model = "modelo-gmm.modelo"

train_data = 'bases/a_b/Treino/X/'
test_data = 'bases/a_b/Teste/X/'

try:
    
    GmM = SR.GMMRec.load(model)

except:
    GmM = SR.GMMRec() # Create a new recognizer
    audio_files = os.listdir(train_data )
    i=0

    while i <len(audio_files):
            f = audio_files[i]
            if (f[-4::] == ".wav"):
                nome,a= f.split('-')           
                (sr, sig) = wav.read(train_data + f)
                novo_mfcc = mfcc(sig,  samplerate=22050, numcep = 13)

    
                GmM.enroll(nome, novo_mfcc)

                                
                                
                    

            i = i+1

    print("training")           
    GmM.train() # Treinar modelo GMM
    print("trained")
    GmM.dump(model) # salvar modelo para verificacao posterior de vozes




acertou = 0


audio_files = os.listdir(test_data )
tamanho = len(audio_files)
i=0
while i <len(audio_files):
    f = audio_files[i]
    nome,a= f.split('-')
    (sr, sig) = wav.read(test_data+f)
    a = mfcc(sig, samplerate=22050, numcep = 13)
    b = GmM.predict(a)[0]
    print(b , nome)
    if(str(b) == str(nome)):
                             
        acertou = acertou +1
    i=i+1





print("acertou: ", acertou,"de: ",tamanho) 
print(acertou/tamanho)
