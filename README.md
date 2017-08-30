# ChemTS
Molecule Design using Monte Carlo Tree Search with Neural Rollout. ChemTS can design novel materials with desired HOMO-LUMO gap and interal energy. Combining with rDock, ChemTS can design molecules active to target proteins.

#  Requirements 
1. [Python](https://www.anaconda.com/download/)>=2.7 
2. [Keras](https://github.com/fchollet/keras)
3. [rdkit](https://anaconda.org/rdkit/rdkit)
4. [rDock](http://rdock.sourceforge.net/installation/)


#  Train a RNN model for molecule generation
1. Run python train_RNN.py to train the RNN model. GPU is highly recommended for reducing the training time.

#  Design materials with desired HOMO-LUMO and internal energy
1. Run python qspr_training.py train a qspr model.
2. Run python qspr_mcts.py to search novel materials with desired HOMO-LUMO gap and energy.

#  Design molecules active to target proteins
1. Run python ligand_mcts.py 

#  MCTS for logP optimization
1. Run python MCTS-RNN.py
