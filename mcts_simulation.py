from subprocess import Popen, PIPE
from math import *
import random
import numpy as np
from copy import deepcopy
from types import IntType, ListType, TupleType, StringTypes
import itertools
import time
import math
import argparse
import subprocess
from load_model import loaded_model
from keras.preprocessing import sequence
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import sys

def chem_kn_simulation(model,state,val):

    end="\n"
    #print state.position
    position=state.position #suppose atom="C"
    total_generated=[]
    new_compound=[]
    get_int_old=[]
    for j in range(len(position)):
        get_int_old.append(val.index(position[j]))

    get_int=get_int_old

    x=np.reshape(get_int,(1,len(get_int)))
    x_pad= sequence.pad_sequences(x, maxlen=82, dtype='int32',
        padding='post', truncating='pre', value=0.)
    while not get_int[-1] == val.index(end):
        predictions=model.predict(x_pad)
        #print predictions.shape
        next_int=np.argmax(predictions[0][len(get_int)-1])
        #print "top 1:",next_int
        a=predictions[0][len(get_int)-1]
        ##next atom probability
        next_int_test=sorted(range(len(a)), key=lambda i: a[i])[-10:]

        #print "top 3:",next_int_test
        get_int.append(next_int)
        #print get_int
        x=np.reshape(get_int,(1,len(get_int)))
        x_pad= sequence.pad_sequences(x, maxlen=82, dtype='int32',
            padding='post', truncating='pre', value=0.)
        if len(get_int)>82:
            break
    total_generated.append(get_int)

    #print total_generated



    return total_generated
