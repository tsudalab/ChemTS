from subprocess import Popen, PIPE
from math import *
import random
import numpy as np
import random as pr
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
from rdkit.Chem import MolFromSmiles, MolToSmiles
import sys

from mcts_simulation import chem_kn_simulation
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
#from unigram import ngram1,ngram2,ngram3,ngram4
from add_node_type import chem_kn_simulation, make_input_smile,predict_smile,check_node_type
from load_model import loaded_activity_model


class chemical:

    def __init__(self):

        self.position=['&']
    def Clone(self):

        st = chemical()
        st.position= self.position[:]
        return st

    def SelectPosition(self,m):

        self.position.append(val2[m])

    def Getatom(self):
        return [i for i in range(self.num_atom)]

class Node:

    def __init__(self, position = None,  parent = None, state = None):
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        self.child=None
        self.wins = 0
        self.visits = 0
        #self.nonvisited_atom=state.Getatom()
        self.type_node=[]
        self.depth=0


    def Selectnode(self):

        #s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + np.sqrt(2)*sqrt(2*log(self.visits)/c.visits))[-1]
        #s=random.choice(self.childNodes)
        ucb=[]
        for i in range(len(self.childNodes)):
        #ind=argmax
            ucb.append(self.childNodes[i].wins/self.childNodes[i].visits+0.01*sqrt(2*log(self.visits)/self.childNodes[i].visits))
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind=pr.choice(indices)
        s=self.childNodes[ind]
        return s

    def Addnode(self, m, s):

        n = Node(position = m, parent = self, state = s)
        self.childNodes.append(n)

    def simulation(self,state):
        predicted_smile=predict_smile(model,state)
        input_smile=make_input_smile(predicted_smile)
        logp,valid_smile,all_smile=logp_calculation(input_smile)

        return logp,valid_smile,all_smile

    def Update(self, result):

        self.visits += 1
        self.wins += result
        #print "self.wins:",self.wins
        #print "self.visits:",self.visits

def MCTS(root, verbose = False):

    """initialization of the chemical trees and grammar trees"""
    run_time=time.time()+600
    rootnode = Node(state = root)
    state = root.Clone()
    maxnum=0
    iteration_num=0
    start_time=time.time()
    """----------------------------------------------------------------------"""


    """global variables used for save valid compounds and simulated compounds"""
    valid_compound=[]
    all_simulated_compound=[]
    desired_compound=[]
    max_score=-100.0
    desired_activity=[]
    time_distribution=[]
    num_searched=[]
    current_score=[]
    depth=[]
    all_score=[]


    """----------------------------------------------------------------------"""

    while time.time()<=run_time:
        iteration_time=time.time()

        node = rootnode # important !    this node is different with state / node is the tree node
        state = root.Clone() # but this state is the state of the initialization .  too important !!!
        """selection step"""
        node_pool=[]
        print "current found max_score:",max_score

        while node.childNodes!=[]:
            node = node.Selectnode()
            state.SelectPosition(node.position)
        print "state position:,",state.position
        #depth.append(len(state.position))


        """------------------------------------------------------------------"""

        """expansion step"""
        all_posible=chem_kn_simulation(model,state.position,val,val2)
        generate_smile=predict_smile(all_posible,val)
        new_compound=make_input_smile(generate_smile)
        node_index,score,valid_smile,all_smile=check_node_type(new_compound,SA_mean,SA_std,logP_mean,logP_std,cycle_mean,cycle_std)
        valid_compound.extend(valid_smile)
        all_simulated_compound.extend(all_smile)
        all_score.extend(score)
        iteration_num=len(all_simulated_compound)
        #if maxnum in [100,200,300,400,500,600,700,800,900,1000]:
         #   used_time=time.time()-start_time
          #  print used_time
           # time_distribution.append(used_time)
        #now_time=time.time()
        #num_searched.append(now_time-start_time)

        if len(node_index)==0:
            re=0.0
            while node != None:
                node.Update(re)
                node = node.parentNode
        else:
            for i in range(len(node_index)):
                m=node_index[i]
                maxnum=maxnum+1
                now_time=time.time()
                if maxnum in [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,
                        2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,
                        4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700,5800,5900,6000,6100,6200,
                        6300,6400,6500,6600,6700,6800,6900,7000,7100,7200,7300,7400,7500,7600,7700,7800,7900,8000,8100,8200,
                        8300,8400,8500,8600,8700,8800,8900,9000,9100,9200,9300,9400,9500,9600,9700,9800,9900,10000]:
                    used_time=time.time()-start_time
                    #print used_time
                    time_distribution.append(used_time)
                num_searched.append(now_time-start_time)
                node.Addnode(m,state)
                node_pool.append(node.childNodes[i])
                #print max_score
                if score[i]>=max_score:
                    max_score=score[i]
                    current_score.append(max_score)
                else:
                    current_score.append(max_score)
                depth.append(len(state.position))
                """simulation"""
                re=1.0/(1.0+np.exp(-score[i]))
                #print re
                """backpropation step"""
            #print "node pool length:",len(node.childNodes)

            for i in range(len(node_pool)):

                node=node_pool[i]
                while node != None:
                    node.Update(re)
                    node = node.parentNode
            finish_iteration_time=time.time()-iteration_time
            print "four step time:",finish_iteration_time

        """check if found the desired compound"""

    #print "all valid compounds:",valid_compound


    #print "logp max found:", current_score
    #print "length of score:",len(current_score)
    #print "time:",time_distribution

    #print "found compounds:",valid_compound
    #print "all valid compounds:", len(valid_compound)
    #print "all compounds:",len(all_simulated_compound)
    #print "all score:", all_score
    print "depth:",depth
    print len(depth)
    print "num_searched=",num_searched


    return valid_compound


def UCTchemical():
    one_search_start_time=time.time()
    time_out=one_search_start_time+60*10
    state = chemical()
    best = MCTS(root = state,verbose = False)


    return best


if __name__ == "__main__":
    smile_old=zinc_data_with_bracket_original()
    val,smile=zinc_processed_with_bracket(smile_old)
    print val
    logP_values = np.loadtxt('logP_values.txt')
    SA_scores = np.loadtxt('SA_scores.txt')
    cycle_scores = np.loadtxt('cycle_scores.txt')
    SA_mean =  np.mean(SA_scores)
    print len(SA_scores)

    SA_std=np.std(SA_scores)
    logP_mean = np.mean(logP_values)
    logP_std= np.std(logP_values)
    cycle_mean = np.mean(cycle_scores)
    cycle_std=np.std(cycle_scores)
    #val2=['C', '(',  'c', '1',  'o', '=', 'O', 'N', 'F', '[C@@H]', 'n',  'S', 'Cl', '[O-]']
    #val2=['C', 'c','#', '3', '(', '2', 'n', 'O', '/', 'N', '=', '\\', ')', '1', 'o', '4', 's', '[C@H]', 'F', 'S', 'Cl', '[C@@H]', '[C@@]', '[C@]', '5', '#', '[nH]', 'Br', 'I', '6', '-', '[NH+]', '[N-]', '[N+]', '[n+]', '[nH+]', '[NH2+]','[NH3+]']
    val2=['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]', 'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/', '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5', '[N-]', '[n+]', '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]', '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]', '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']

    model=loaded_model()
    #acitivity_model=loaded_activity_model()
    valid_compound=UCTchemical()
