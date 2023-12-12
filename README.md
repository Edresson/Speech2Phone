# Speech2Phone
This is the official implementation of the  paper *Speech2Phone: A new and efficient method for training speaker recognition models*

In this repository the "Paper" directory has the implementation of all the experiments and topologies explored in the article.
  The Speech2Phone directory presents the implementation and checkpoints of the best model of the article.


## Datasets download links:
[Speech2Phone Dataset V1](https://drive.google.com/uc?id=1jiL7uL5zHp4i6pO14jqqe2uzX_IxPJkE&export=download)

[Common Voice ZH (TW)](https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/zh-HK.tar.gz)

[Common Voice PT](https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/pt.tar.gz)

[Common Voice ES](https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/es.tar.gz)

[VCTK removed silence](https://www.dropbox.com/s/9n8sd97qvjijqa1/VCTK-Corpus-Removed-Silence.zip?dl=0)

## Colab Notebook Demos:

     [Identification of speakers in Spanish](https://colab.research.google.com/drive/1POsM0G7F-sZRHRp6bJt4Ym3rzVn-EcyU)

     [Identification of speakers in Chinese spoken in Taiwan](https://colab.research.google.com/drive/1PV4FTQDhNIu1BZKrF3Ehe1VY8LgGK-0i)


## Citation

```
@InProceedings{10.1007/978-3-030-91699-2_39,
author="Casanova, Edresson
and Candido Junior, Arnaldo
and Shulby, Christopher
and de Oliveira, Frederico Santos
and Gris, Lucas Rafael Stefanel
and da Silva, Hamilton Pereira
and Alu{\'i}sio, Sandra Maria
and Ponti, Moacir Antonelli",
editor="Britto, Andr{\'e}
and Valdivia Delgado, Karina",
title="Speech2Phone: A Novel and Efficient Method for Training Speaker Recognition Models",
booktitle="Intelligent Systems",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="572--585",
abstract="In this paper we present an efficient method for training models for speaker recognition using small or under-resourced datasets. This method requires less data than other SOTA (State-Of-The-Art) methods, e.g. the Angular Prototypical and GE2E loss functions, while achieving similar results to those methods. This is done using the knowledge of the reconstruction of a phoneme in the speaker's voice. For this purpose, a new dataset was built, composed of 40 male speakers, who read sentences in Portuguese, totaling approximately 3h. We compare the three best architectures trained using our method to select the best one, which is the one with a shallow architecture. Then, we compared this model with the SOTA method for the speaker recognition task: the Fast ResNet--34 trained with approximately 2,000 h, using the loss functions Angular Prototypical and GE2E. Three experiments were carried out with datasets in different languages. Among these three experiments, our model achieved the second best result in two experiments and the best result in one of them. This highlights the importance of our method, which proved to be a great competitor to SOTA speaker recognition models, with 500x less data and a simpler approach.",
isbn="978-3-030-91699-2"
}

```
