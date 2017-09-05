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
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
from add_node_type import chem_kn_simulation, make_input_smile,predict_smile,check_node_type,node_to_add,expanded_node


class chemical:

    def __init__(self):

        self.position=['&']
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
        #self.nonvisited_atom=state.Getatom()
        #self.type_node=tp
        self.depth=0


    def Selectnode(self):

        #s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + np.sqrt(2)*sqrt(2*log(self.visits)/c.visits))[-1]
        #s=random.choice(self.childNodes)
        ucb=[]
        for i in range(len(self.childNodes)):
        #ind=argmax
            ucb.append(self.childNodes[i].wins/self.childNodes[i].visits+1.0*sqrt(2*log(self.visits)/self.childNodes[i].visits))
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
    run_time=time.time()+600*2
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

    while maxnum<10100:
        print maxnum
        #iteration_time=time.time()

        node = rootnode # important !    this node is different with state / node is the tree node
        state = root.Clone() # but this state is the state of the initialization .  too important !!!
        """selection step"""
        node_pool=[]
        print "current found max_score:",max_score

        while node.childNodes!=[]:
            node = node.Selectnode()
            state.SelectPosition(node.position)
        print "state position:,",state.position
        depth.append(len(state.position))


        """------------------------------------------------------------------"""

        """expansion step"""
        """calculate how many nodes will be added under current leaf"""
        expanded=expanded_node(model,state.position,val)
        nodeadded=node_to_add(expanded,val)


        all_posible=chem_kn_simulation(model,state.position,val,nodeadded)
        generate_smile=predict_smile(all_posible,val)
        new_compound=make_input_smile(generate_smile)

        node_index,score,valid_smile,all_smile=check_node_type(new_compound,SA_mean,SA_std,logP_mean,logP_std,cycle_mean,cycle_std)

        print node_index
        valid_compound.extend(valid_smile)
        all_simulated_compound.extend(all_smile)
        all_score.extend(score)
        iteration_num=len(all_simulated_compound)
        if len(node_index)==0:
            re=-1.0
            while node != None:
                node.Update(re)
                node = node.parentNode
        else:
            for i in range(len(node_index)):
                #m=node_index[i]
                maxnum=maxnum+1
                node.Addnode(nodeadded[i],state)
                node_pool.append(node.childNodes[i])
                if score[i]>=max_score:
                    max_score=score[i]
                    current_score.append(max_score)
                else:
                   current_score.append(max_score)
                depth.append(len(state.position))
                """simulation"""
                re=(0.8*score[i])/(1.0+abs(0.8*score[i]))
                if maxnum==100:
                    maxscore100=max_score
                    time100=time.time()-start_time
                if maxnum==500:
                    maxscore500=max_score
                    time500=time.time()-start_time
                if maxnum==1000:

                    maxscore1000=max_score
                    time1000=time.time()-start_time
                if maxnum==5000:
                    maxscore5000=max_score
                    time5000=time.time()-start_time
                if maxnum==10000:
                    time10000=time.time()-start_time
                    maxscore10000=max_score
                    #valid10000=10000*1.0/len(all_simulated_compound)
                """backpropation step"""
            #print "node pool length:",len(node.childNodes)

            for i in range(len(node_pool)):

                node=node_pool[i]
                while node != None:
                    node.Update(re)
                    node = node.parentNode
            #finish_iteration_time=time.time()-iteration_time
            #print "four step time:",finish_iteration_time

        """check if found the desired compound"""

    #print "all valid compounds:",valid_compound

    finished_run_time=time.time()-start_time

    print "logp max found:", current_score
    #print "length of score:",len(current_score)
    #print "time:",time_distribution

    print "valid_com=",valid_compound
    print "num_valid:", len(valid_compound)
    print "all compounds:",len(all_simulated_compound)
    print "score=", all_score
    print "depth=",depth
    print len(depth)
    print "runtime",finished_run_time
    #print "num_searched=",num_searched
    print "100 max:",maxscore100,time100
    print "500 max:",maxscore500,time500
    print "1000 max:",maxscore1000,time1000
    print "5000 max:",maxscore5000,time5000
    print "10000 max:",maxscore10000,time10000
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
    acitivity_model=loaded_activity_model()
    valid_compound=UCTchemical()
