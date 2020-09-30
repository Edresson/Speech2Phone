# -*- coding: utf-8 -*-

import os

import string
import os
import importlib
import random
import librosa
import sys
import torch


import numpy as np
from tqdm import tqdm

'''# downlaod vctk corpus
if not os.path.isdir('VCTK-Corpus-removed-silence'):
  !wget https://www.dropbox.com/s/9n8sd97qvjijqa1/VCTK-Corpus-Removed-Silence.zip?dl=0 -O VCTK-Corpus-Removed-Silence.zip
  !unzip VCTK-Corpus-Removed-Silence.zip

!pip install pydub tensorflow==1.14.0 tflearn==0.3.2

#Download Speech2Phone Checkpoint
!wget -O ./saver.zip https://www.dropbox.com/s/b19xt2wu3th9p36/Save-Models-Speaker-Diarization.zip?dl=0
!mkdir Speech2Phone
!unzip saver.zip
!mv  Save-Models/  Speech2Phone/Save-Models/
'''
#Utils for Speech2Phone Preprocessing
from pydub import AudioSegment as audio

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms
 
    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0  # ms
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        #print(trim_ms,len(sound))
        if trim_ms > len(sound):
            return None
        trim_ms += chunk_size
 
    return trim_ms

def remove_silence(sound):
    start_trim = detect_leading_silence(sound)
    if start_trim is None:
        return None
    end_trim = detect_leading_silence(sound.reverse())
    duration = len(sound)
    trimmed_sound = sound[start_trim:duration-end_trim]
    return trimmed_sound

import tflearn

#Create model for restore
encoder = tflearn.input_data(shape=[None, 13,int(216)])
encoder = tflearn.dropout(encoder,0.9) #10 % drop - 90% -> 80
encoder = tflearn.dropout(encoder,0.2)# 80 % drop
encoder = tflearn.fully_connected(encoder, 40,activation='crelu')
decoder = tflearn.fully_connected(encoder, int(572), activation='linear')
net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.0007,loss='mean_square', metric=None)#categorical_crossentropy
model = tflearn.DNN(net, tensorboard_verbose=0,tensorboard_dir='tflearn_logs')

model.load('Save-Models/Model3-Best-40loc.tflearn')

encoding_model = tflearn.DNN(encoder, session=model.session)# used for extract embedding in encoder layer

# make metadat with vctk corpus
meta_data = []
vctk_dir = 'VCTK-Corpus-removed-silence/wav48/'
speakers = os.listdir(vctk_dir)
for speaker in speakers:
  samples = os.listdir(os.path.join(vctk_dir, speaker))
  for sample in samples:
    file_dir = os.path.join(vctk_dir, speaker, sample)
    meta_data.append(file_dir)

#Preprocess dataset
embeddings_dict = {}
len_meta_data = len(meta_data)
for i in tqdm(range(len_meta_data)):
    wave_file_path = meta_data[i]
    speaker_id = wave_file_path
    try:
        sound = audio.from_wav(wave_file_path)
    except:
        print("erro ler arquivo")
        continue
    wave = remove_silence(sound)
    if wave is None:
        print("erro remove silence")
        continue
    
    file_embeddings = None
    begin = 0
    end = 5
    step = 1 
    if int(wave.duration_seconds) < 5: # 5 seconds is the Speech2Phone input if is small concate
        aux = wave
        while int(aux.duration_seconds) <= 5:
            aux += wave
        wave = aux
        del aux
        
    while (end) <= int(wave.duration_seconds):
        try:        
            segment = wave[begin*1000:end*1000]
            segment.export('../aux' + '.wav', 'wav')# its necessary because pydub and librosa load wave in diferent form 
            y, sr = librosa.load('../aux.wav',sr=22050)#sample rate = 22050 

            if file_embeddings is None:
                file_embeddings =[np.array(encoding_model.predict([librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)])[0])]
            else:
                file_embeddings.append(np.array(encoding_model.predict([librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)])[0]))   
            os.system('rm ../aux.wav')
            begin = begin + step
            end = end + step
        except Exception as e:
            print(e)
            print('parte do arquivo deu erro', len(file_embeddings))
            begin = begin + step
            end = end + step
    embeddings_dict[wave_file_path] = torch.nn.functional.normalize(torch.FloatTensor(np.mean(np.array(file_embeddings), axis=0).reshape(1, -1).tolist()), p=2, dim=1)
    del file_embeddings

#check emb dim
for key in embeddings_dict.keys():
  print(embeddings_dict[key].shape)
  break

emb_file = 'embeddings/vctk-speech2phone-embeddings.pt'
torch.save(embeddings_dict, open(emb_file, "wb"))