# -*- coding: utf-8 -*-



import os

from sklearn import metrics
import numpy, torch, torch.nn.functional as F
import time, sys
import random


problem_files = []
def calculate_score(ref_feat, com_feat, num_eval, normalize=True):
        if normalize:
            ref_feat = F.normalize(ref_feat, p=2, dim=1)
            com_feat = F.normalize(com_feat, p=2, dim=1)

        dist = F.pairwise_distance(ref_feat.unsqueeze(-1).expand(-1,-1,num_eval), com_feat.unsqueeze(-1).expand(-1,-1,num_eval).transpose(0,2)).detach().cpu().numpy();

        score = -1 * numpy.mean(dist)

        return score

def evaluateFromEmbeddings(listfilename, embeddings_file, num_eval, print_interval=5000, normalize=True):
    
    lines       = []
    files       = []
    filedict    = {}
    tstart      = time.time()

    ## Read all lines
    with open(listfilename) as listfile:
        while True:
            line = listfile.readline();
            if (not line): #  or (len(all_scores)==1000) 
                break;

            data = line.split();
            # ignore head
            if 'class' in data[0]:
              continue
            files.append(data[1])
            files.append(data[2])
            lines.append(line)

    setfiles = list(set(files))
    setfiles.sort()

    feats = torch.load(embeddings_file)

    all_scores = [];
    all_labels = [];
    tstart = time.time()
    #lines = lines[:20]
    ## Read files and compute all scores
    for idx, line in enumerate(lines):

        data = line.split();
        try:
          
          ref_feat = feats[data[1]].cuda()
          # pt/clips/common_voice_pt_20143235.mp3-22k.wav
          #pt/clips/common_voice_pt_20081091.mp3-22k.wav
          com_feat = feats[data[2]].cuda()
        except Exception as e: # if one sample dont exist ignore the pair
          #print(feats.keys())
          print(list(feats.keys())[0], e)
          #problem_files.append(e)
          #print("Erro: ",e)
          #print('ignore', data[1], data[2])
          exit()
          continue

        score = calculate_score(ref_feat, com_feat, num_eval, normalize)

        all_scores.append(score);  
        all_labels.append(int(data[0]));

        if idx % print_interval == 0:
            telapsed = time.time() - tstart
            sys.stdout.write("\rComputing %d: %.2f Hz"%(idx,idx/telapsed));
            sys.stdout.flush();

    print('\n')

    return (all_scores, all_labels);

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    fnr = fnr*100
    fpr = fpr*100

    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])
    
    return (tunedThreshold, eer, fpr, fnr);





out_csv_file = "./veri_vctk.csv"

test_list = out_csv_file
num_eval = 1

# get torch embedding files, for this you need run the notebook: VCTK-Speech2Phone-ExtractSpeakerEmbeddings.ipynb ( https://colab.research.google.com/drive/1QnKm_0vSySWwD5Y56xLpDFUWY4HNSX3q?usp=sharing)
emb_file = 'embeddings/vctk-speech2phone-embeddings.pt' 
sc, lab = evaluateFromEmbeddings(listfilename=test_list, embeddings_file=emb_file, num_eval=num_eval, print_interval=100, normalize=False)
result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
print('VCTK Speech2Phone EER %2.4f'%result[1])



# get torch embedding files, for this you need run the notebook: VCTK-Speech2Phone-ExtractSpeakerEmbeddings.ipynb ( https://colab.research.google.com/drive/1QnKm_0vSySWwD5Y56xLpDFUWY4HNSX3q?usp=sharing)
emb_file = 'embeddings/common-voice-pt-speech2phone-embeddings.pt' #speech2phone
#num_eval = 1 #speech2phone
sc, lab = evaluateFromEmbeddings(listfilename=test_list, embeddings_file=emb_file, num_eval=num_eval, print_interval=100, normalize=True)
result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
print('Common Voice PT Speech2Phone EER %2.4f'%result[1])


problem_files = []
out_csv_file = "./Common_Voice/cv-corpus-5.1-2020-06-22/veri_chines-common_voice-22khz.csv"
test_list = out_csv_file
num_eval = 1 
# get torch embedding files, for this you need run the notebook: VCTK-Speech2Phone-ExtractSpeakerEmbeddings.ipynb ( https://colab.research.google.com/drive/1QnKm_0vSySWwD5Y56xLpDFUWY4HNSX3q?usp=sharing)
emb_file = 'embeddings/common-voice-chines-speech2phone-embeddings.pt' #speech2phone
#num_eval = 1 #speech2phone
sc, lab = evaluateFromEmbeddings(listfilename=test_list, embeddings_file=emb_file, num_eval=num_eval, print_interval=100, normalize=True)
result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
print('Common Voice ZH Speech2Phone EER %2.4f'%result[1])



problem_files = []
out_csv_file = "./Common_Voice/cv-corpus-5.1-2020-06-22/veri_es-common_voice-22khz.csv"
test_list = out_csv_file
num_eval = 1 

# get torch embedding files, for this you need run the notebook: VCTK-Speech2Phone-ExtractSpeakerEmbeddings.ipynb ( https://colab.research.google.com/drive/1QnKm_0vSySWwD5Y56xLpDFUWY4HNSX3q?usp=sharing)
emb_file = 'embeddings/common-voice-es-speech2phone-embeddings.pt' #speech2phone
sc, lab = evaluateFromEmbeddings(listfilename=test_list, embeddings_file=emb_file, num_eval=num_eval, print_interval=100, normalize=True)

result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
print('Common Voice ES Speech2Phone EER %2.4f'%result[1])



