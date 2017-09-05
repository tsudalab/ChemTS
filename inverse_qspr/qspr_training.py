from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np
import csv
import itertools
import operator
import numpy as np
import nltk
import os
import sys
from datetime import datetime
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers import Dropout

from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
import json

def organic():

    sen_space=[]
    #f = open('/Users/yang/smiles.csv', 'rb')
    f = open('data/smile_trainning.csv', 'rb')
    #f = open('/Users/yang/Downloads/molecule-autoencoder-master/data/250k_rndm_zinc_drugs_clean.smi', 'rb')

    reader = csv.reader(f)
    for row in reader:
        #word_space[row].append(reader[row])
        #print word_sapce
        sen_space.append(row)
    #print sen_space
    f.close()

    #element_table=["Cu","Ti","Zr","Ga","Ge","As","Se","Br","Si","Zn","Cl","Be","Ca","Na","Sr","Ir","Li","Rb","Cs","Fr","Be","Mg",
    #        "Ca","Sr","Ba","Ra","Sc","La","Ac","Ti","Zr","Nb","Ta","Db","Cr","Mo","Sg","Mn","Tc","Re","Bh","Fe","Ru","Os","Hs","Co","Rh",
    #        "Ir","Mt","Ni","Pd","Pt","Ds","Cu","Ag","Au","Rg","Zn","Cd","Hg","Cn","Al","Ga","In","Tl","Nh","Si","Ge","Sn","Pb","Fl",
    #        "As","Sb","Bi","Mc","Se","Te","Po","Lv","Cl","Br","At","Ts","He","Ne","Ar","Kr","Xe","Rn","Og"]
    element_table=["C","N","O","P","S","F","Cl","Br","I","(",")","=","#"]
    #print sen_space
    word1=sen_space[0]
    word_space=list(word1[0])
    end="\n"
    #print word1
    #print word_space
    all_smile=[]
    organic_smile=[]
    t=0
    #print sen_space

    while t <len(sen_space):
        #print t
        word1=sen_space[t]
        word_space=list(word1[0])
        word=[]
        #if "[" not in word_space:
        organic_smile.append(word_space)

        t=t+1

    #print len(organic_smile)
    #print len(organic_smile)
    #print organic_smile

    return organic_smile






def process_organic(sen_space):
    #print sen_space
    all_smile=[]
    end="\n"
    element_table=["C","N","B","O","P","S","F","Cl","Br","I","(",")","=","#"]
    ring=["1","2","3","4","5","6","7","8","9","10"]

    for i in range(len(sen_space)):
        #word1=sen_space[i]
        word_space=sen_space[i]
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

        all_smile.append(list(word))

    val=["\n"]

    for i in range(len(all_smile)):
        for j in range(len(all_smile[i])):
            if all_smile[i][j] not in val:
                val.append(all_smile[i][j])

    processed_smile=[]
    for i in range(len(all_smile)):
        middle=[]
        for j in range(len(all_smile[i])):
            middle.append(all_smile[i][j])
        com=''.join(middle)
        processed_smile.append(com)

    return  processed_smile



def prepare_data(smile):

    #for i in range(len)
    X_train=[]
    for i in range(len(smile)):
        m = Chem.MolFromSmiles(smile[i])
        if m !=None:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(m,2,1024)
            a=list(fp1.ToBitString())
            X_train.append(a)
        else:
            print i
            print smile[i]

    X_train_final=np.matrix(X_train)
    print X_train_final.shape

    return X_train_final


def prepare_y_data():
    reader = csv.reader(open("data/yoshida_y_train.csv", "rb"), delimiter=",")
    y = list(reader)

    result = np.array(y).astype('float')
    #print result.shape
    arr = np.delete(result, 16538, axis=0)
    return arr


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("homo_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("homo_model.h5")
    print("Saved model to disk")



def qspr_loaded_homo_model():
    # load json and create model
    json_file = open("homo_model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("homo_model.h5")
    print("Loaded model from disk")

    return loaded_model





if __name__ == "__main__":
    y1=prepare_y_data()

    yoshida_data=organic()
    smile=process_organic(yoshida_data)
    X1=prepare_data(smile)
    idx=np.random.permutation(16673)
    X=X1[idx]
    y=y1[idx]
    X_train=X[0:10000,:]
    y_train=y[0:10000,:]
    X_test=X[10000:,:]
    y_test=y[10000:16673,:]
    with open('energy_y_trian.txt', 'w') as f:
        json.dump(y_train[0:10000,1].tolist(), f)
    with open('homo_y_trian.txt', 'w') as f:
        json.dump(y_train[0:10000,0].tolist(), f)

    with open('energy_y_test.txt', 'w') as f:
        json.dump(y_test[10000:16673,1].tolist(), f)
    with open('homo_y_test.txt', 'w') as f:
        json.dump(y_train[10000:16673,0].tolist(), f)

    model = Sequential()
    model.add(Dense(512, input_dim=1024, kernel_initializer='normal', activation='elu'))

    model.add(Dense(512, kernel_initializer='normal', activation='elu'))

    model.add(Dense(2, kernel_initializer='normal'))

    print(model.summary())
    model.compile(loss='mean_squared_error', optimizer="Adam",metrics=["mae"])

    model.fit(X_train, y_train,nb_epoch=200, batch_size=32,validation_data=(X_test,y_test))
    #loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32)
    #predictions = model.predict(X_test)
    save_model(model)
    #pr=np.sum(abs(y_test-predictions))/4673.0
    #### load model
    model=qspr_loaded_homo_model()
    model.compile(loss='mae', optimizer='adam',metrics=["mse"])

    preds = model.predict(X_test)
    preds_tr=model.predict(X_train)
