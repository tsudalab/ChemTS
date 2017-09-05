from subprocess import Popen, PIPE
from math import *
import random
import random as pr
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
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
from add_node_type import chem_kn_simulation, make_input_smile,predict_smile,check_node_type,node_to_add,expanded_node



class chemical:

    def __init__(self):

        self.position=['&']
        self.num_atom=8
        self.vl=['\n', '&', 'C', '(', 'c', '1', 'o', '=', 'O', 'N', 'F', '[C@@H]',
        'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/', '[nH]', '[NH3+]',
     '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '[N-]', '[n+]', '[S@@]', '[S-]',
         'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]', '[PH2]', '[P@]', '[P+]', '[S+]',
        '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]', '[s+]', '[PH+]', '[PH]', '[S@@+]']

    def Clone(self):

        st = chemical()
        st.position= self.position[:]
        return st

    def SelectPosition(self,m):
        self.position.append(m)

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
        self.nonvisited_atom=state.Getatom()
        self.type_node=[]
        self.depth=0


    def Selectnode(self):

        #s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + 0.8*sqrt(2*log(self.visits)/c.visits))[-1]
        #s=random.choice(self.childNodes)
        ucb=[]
        for i in range(len(self.childNodes)):
            ucb.append(self.childNodes[i].wins/self.childNodes[i].visits+sqrt(2)*sqrt(2*log(self.visits)/self.childNodes[i].visits))
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


def MCTS(root, verbose = False):

    """initialization of the chemical trees and grammar trees"""
    run_time=time.time()+3600*48
    rootnode = Node(state = root)
    state = root.Clone()
    """----------------------------------------------------------------------"""


    """global variables used for save valid compounds and simulated compounds"""
    valid_compound=[]
    all_simulated_compound=[]
    desired_compound=[]
    max_logp=[]
    desired_activity=[]
    depth=[]
    min_score=1000
    score_distribution=[]
    min_score_distribution=[]

    """----------------------------------------------------------------------"""

    while time.time()<=run_time:

        node = rootnode # important !    this node is different with state / node is the tree node
        state = root.Clone() # but this state is the state of the initialization .  too important !!!
        """selection step"""
        node_pool=[]

        while node.childNodes!=[]:
            node = node.Selectnode()
            state.SelectPosition(node.position)
        print "state position:,",state.position


        """------------------------------------------------------------------"""

        """expansion step"""
        expanded=expanded_node(model,state.position,val)
        nodeadded=node_to_add(expanded,val)
        all_posible=chem_kn_simulation(model,state.position,val,nodeadded)
        generate_smile=predict_smile(all_posible,val)
        new_compound=make_input_smile(generate_smile)



        node_index,rdock_score,valid_smile=check_node_type(new_compound)
        valid_compound.extend(valid_smile)
        score_distribution.extend(rdock_score)

        if len(node_index)==0:
            re=-1.0
            while node != None:
                node.Update(re)
                node = node.parentNode
        else:
            for i in range(len(node_index)):
                m=node_index[i]
                node.Addnode(nodeadded[i],state)
                node_pool.append(node.childNodes[i])
                depth.append(len(state.position))
                print "current minmum score",min_score
                if rdock_score[i]<=min_score:
                    min_score_distribution.append(rdock_score[i])
                    min_score=rdock_score[i]
                else:
                    min_score_distribution.append(min_score)
                """simulation"""
                re=(-0.8*rdock_score[i])/(1+0.8*abs(rdock_score[i]))
                """backpropation step"""

            for i in range(len(node_pool)):

                node=node_pool[i]
                while node != None:
                    node.Update(re)
                    node = node.parentNode




        """check if found the desired compound"""

    #print "all valid compounds:",valid_compound
    #print "all active compounds:",desired_compound
    print "rdock_score",score_distribution
    print "num valid_compound:",len(valid_compound)
    print "valid compounds",valid_compound
    print "depth",depth
    print "min_score",min_score_distribution


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


    model=loaded_model()
    valid_compound=UCTchemical()
