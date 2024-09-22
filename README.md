This repository contains the materials produced in the context of C. Grimal's Masters Thesis at University of Essex.
The aim of our research was to build a transfer learning framework in order to apply English knowledge gained on a sarcasm detection classification task to French data.
All content is non-commercial in nature. Everything interesting can be found in the Binaries folder, everything else is old material kept only for the purpose of archiving. 

Data files (to heavy for github) can be found at https://mega.nz/folder/yaQlVbyY#WH--WdWeIAgR1eLA86KR5w
The complete SARC dataset initially used can be found at https://nlp.cs.princeton.edu/old/SARC/2.0/

Disregard all files aside from reduced_1000.csv and/or translated_r2500.csv, and optionally the word vectors if do not plan on generating your own. 
Do note that in the case of the French model, fasttext_fr.bin pre-trained embeddings are mandatory, unless you want to build your own.

Find runner.sh executable in relevant folder 

Reminder to run things in the folder they were found, if you want them to work

Reminder to install requirements.txt, again, if you want things to work

scipy must not be over version 10.10.1 for cause of deprecated functions still being used by gensim

SARC dataset published by Mikhail Khodak,
@unpublished{SARC,
  authors={Mikhail Khodak and Nikunj Saunshi and Kiran Vodrahalli},
  title={A Large Self-Annotated Corpus for Sarcasm},
  url={https://arxiv.org/abs/1704.05579},
  year=2017
}
