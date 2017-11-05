import csv
import itertools
import operator
import numpy as np
import nltk
import os
import sys
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
#from make_smile import ziprocess_organic,process_zinc_data
from make_smile import zinc_data_with_bracket_original,zinc_processed_with_bracket

from keras.layers import Conv1D, MaxPooling1D
#from combine_bond_atom import organic, process_organic,bond_atom

def load_data():

    sen_space=[]
    f = open('/Users/yang/smiles.csv', 'rb')
    reader = csv.reader(f)
    for row in reader:
        #word_space[row].append(reader[row])
        #print word_sapce
        sen_space.append(row)
    #print sen_space
    f.close()

    element_table=["Cu","Ti","Zr","Ga","Ge","As","Se","Br","Si","Zn","Cl","Be","Ca","Na","Sr","Ir","Li","Rb","Cs","Fr","Be","Mg",
            "Ca","Sr","Ba","Ra","Sc","La","Ac","Ti","Zr","Nb","Ta","Db","Cr","Mo","Sg","Mn","Tc","Re","Bh","Fe","Ru","Os","Hs","Co","Rh",
            "Ir","Mt","Ni","Pd","Pt","Ds","Cu","Ag","Au","Rg","Zn","Cd","Hg","Cn","Al","Ga","In","Tl","Nh","Si","Ge","Sn","Pb","Fl",
            "As","Sb","Bi","Mc","Se","Te","Po","Lv","Cl","Br","At","Ts","He","Ne","Ar","Kr","Xe","Rn","Og"]
    #print sen_space
    word1=sen_space[0]
    word_space=list(word1[0])
    end="\n"
    #start="st"
    #word_space.insert(0,end)
    word_space.append(end)
    #print word_space
    #print len(sen_space)
    all_smile=[]
    #print word_space
    #for i in range(len(all_smile)):

    for i in range(len(sen_space)):
        word1=sen_space[i]
        word_space=list(word1[0])
        word=[]
        #word_space.insert(0,end)
        j=0
        while j<len(word_space):
            word_space1=[]
            word_space1.append(word_space[j])
            if j+1<len(word_space):
                word_space1.append(word_space[j+1])
                word_space2=''.join(word_space1)
            else:
                word_space1.insert(0,word_space[j-1])
                word_space2=''.join(word_space1)
            if word_space2 not in element_table:
                word.append(word_space[j])
                j=j+1
            else:
                word.append(word_space2)
                j=j+2

        word.append(end)
        all_smile.append(list(word))
    #print all_smile
    val=[]
    for i in range(len(all_smile)):
        for j in range(len(all_smile[i])):
            if all_smile[i][j] not in val:
                val.append(all_smile[i][j])
    #print val
    val.remove("\n")
    val.insert(0,"\n")

    return val, all_smile


def organic_data():
    sen_space=[]
    #f = open('/Users/yang/smiles.csv', 'rb')
    f = open('/Users/yang/LSTM-chemical-project/make_sm.csv', 'rb')
    #f = open('/Users/yang/LSTM-chemical-project/smile_trainning.csv', 'rb')
    reader = csv.reader(f)
    for row in reader:
        #word_space[row].append(reader[row])
        #print word_sapce
        sen_space.append(row)
    #print sen_space
    f.close()

    element_table=["Cu","Ti","Zr","Ga","Ge","As","Se","Br","Si","Zn","Cl","Be","Ca","Na","Sr","Ir","Li","Rb","Cs","Fr","Be","Mg",
            "Ca","Sr","Ba","Ra","Sc","La","Ac","Ti","Zr","Nb","Ta","Db","Cr","Mo","Sg","Mn","Tc","Re","Bh","Fe","Ru","Os","Hs","Co","Rh",
            "Ir","Mt","Ni","Pd","Pt","Ds","Cu","Ag","Au","Rg","Zn","Cd","Hg","Cn","Al","Ga","In","Tl","Nh","Si","Ge","Sn","Pb","Fl",
            "As","Sb","Bi","Mc","Se","Te","Po","Lv","Cl","Br","At","Ts","He","Ne","Ar","Kr","Xe","Rn","Og"]
    #print sen_space
    word1=sen_space[0]
    word_space=list(word1[0])
    end="\n"
    #start="st"
    #word_space.insert(0,end)
    word_space.append(end)
    #print word_space
    #print len(sen_space)
    all_smile=[]
    #print word_space
    #for i in range(len(all_smile)):

    for i in range(len(sen_space)):
        word1=sen_space[i]
        word_space=list(word1[0])
        word=[]
        #word_space.insert(0,end)
        j=0
        while j<len(word_space):
            word_space1=[]
            word_space1.append(word_space[j])
            if j+1<len(word_space):
                word_space1.append(word_space[j+1])
                word_space2=''.join(word_space1)
            else:
                word_space1.insert(0,word_space[j-1])
                word_space2=''.join(word_space1)
            if word_space2 not in element_table:
                word.append(word_space[j])
                j=j+1
            else:
                word.append(word_space2)
                j=j+2

        word.append(end)
        all_smile.append(list(word))
    #print all_smile
    val=[]
    for i in range(len(all_smile)):
        for j in range(len(all_smile[i])):
            if all_smile[i][j] not in val:
                val.append(all_smile[i][j])
    #print val
    val.remove("\n")
    val.insert(0,"\n")

    return val, all_smile


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


def generate_smile(model,val):
    end="\n"
    start_smile_index= [val.index("C")]
    new_smile=[]

    while not start_smile_index[-1] == val.index(end):
        predictions=model.predict(start_smile_index)
        ##next atom probability
        smf=[]
        for i in range (len(X)):
            sm=[]
            for j in range(len(X[i])):
                #if np.argmax(predictions[i][j])=!0
                sm.append(np.argmax(predictions[i][j]))
            smf.append(sm)

        #print sm
        #print smf
        #print len(sm)

        new_smile.append(sampled_word)
    #sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    #return new_sentence



def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

if __name__ == "__main__":
    smile=zinc_data_with_bracket_original()
    valcabulary,all_smile=zinc_processed_with_bracket(smile)
    print(valcabulary)
    print(len(all_smile))
    X_train,y_train=prepare_data(valcabulary,all_smile)
  
    maxlen=81


    X= sequence.pad_sequences(X_train, maxlen=81, dtype='int32',
        padding='post', truncating='pre', value=0.)
    y = sequence.pad_sequences(y_train, maxlen=81, dtype='int32',
        padding='post', truncating='pre', value=0.)
    
    
    y_train_one_hot = np.array([to_categorical(sent_label, num_classes=len(valcabulary)) for sent_label in y])
    print (y_train_one_hot.shape)

    vocab_size=len(valcabulary)
    embed_size=len(valcabulary)

    
    N=X.shape[1]


    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=len(valcabulary), input_length=N,mask_zero=False))
    model.add(GRU(output_dim=256, input_shape=(81,64),activation='tanh',return_sequences=True))
    #model.add(LSTM(output_dim=256, input_shape=(81,64),activation='tanh',return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(256,activation='tanh',return_sequences=True))
    #model.add(LSTM(output_dim=1000, activation='sigmoid',return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(embed_size, activation='softmax')))
    optimizer=Adam(lr=0.01)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X,y_train_one_hot,nb_epoch=100, batch_size=512,validation_split=0.1)
    save_model(model)
