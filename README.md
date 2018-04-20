# NLU3
NLU Assignment 3

Please use python3 for compatibility.

Mallet should be downloaded into the same folder as git repo folder with folder name `mallet`
Please check the requirements.txt and install the following packages.

nltk - `pip3 install --user nltk`

gensim - `pip3 install --user gensim`

tensorflow - `pip install --user tensorflow`

keras - `pip3 install --user keras`

keras_contrib - `pip3 install --user git+https://www.github.com/keras-team/keras-contrib.git`

Instructions :

    Create splits, models and output folders and copy mallet if not present.
    Execute create_data_splits.py to create the data splits required for mallet.
    Execute sh train_mallet.sh first to create the models.
    Execute sh get_scores.sh to get the test scores.
    Execute python3 sequential_NER.py to run the Bi-LSTM with CRF model.

Log files : 

    get_scores.log : logs running from get_scores.sh
    sequential_NER.log : logs from running sequential_NER.py
