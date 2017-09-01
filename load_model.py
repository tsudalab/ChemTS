import csv
import itertools
import operator
import numpy as np
import nltk
import h5py
import os
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation,TimeDistributed
from keras.layers import LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers import Dropout
import numpy as np
import random
import sys
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import model_from_json
from make_smile import zinc_data_with_bracket, zinc_processed_with_bracket

def prepare_data(smiles,all_smile):
    all_smile_index=[]
    for i in range(len(all_smile)):
        smile_index=[]
        for j in range(len(all_smile[i])):
            smile_index.append(smiles.index(all_smile[i][j]))
        all_smile_index.append(smile_index)
    X_train=all_smile_index
    y_train=[]
    for i in range(len(X_train)):

        x1=X_train[i]
        x2=x1[1:len(x1)]
        x2.append(0)
        y_train.append(x2)

    return X_train,y_train


def loaded_model():

    json_file = open('RNN-model/model.json', 'r')
    #json_file = open('/Users/yang/LSTM-chemical-project/protein-ligand/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    #loaded_model.load_weights('/Users/yang/LSTM-chemical-project/protein-ligand/model.h5')
    loaded_model.load_weights('RNN-model/model.h5')
    print("Loaded model from disk")
    

    return loaded_model

def loaded_activity_model():
    #smile_old=zinc_data_with_bracket_original()
    #valcabulary,smile=zinc_processed_with_bracket(smile_old)

    #json_file = open('/Users/yang/LSTM-chemical-project/new-version-rnn/model/model.json', 'r')
    json_file = open('/Users/yang/LSTM-chemical-project/protein-ligand/ppara_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights('/Users/yang/LSTM-chemical-project/protein-ligand/ppara_model.h5')
    #loaded_model.load_weights('/Users/yang/LSTM-chemical-project/new-version-rnn/model/model.h5')
    print("Loaded model from disk")

    return loaded_model

loaded_model()
