# ChemTS
Molecule Design using Monte Carlo Tree Search with Neural Rollout. ChemTS can design novel materials with desired HOMO-LUMO gap and interal energy. Combining with rDock, ChemTS can design molecules active to target proteins.

#  Requirements 
1. Python>=2.7
2. Keras library
3. Rdkit
4. rDock


#  Train a RNN model for molecule generation
Run python train_RNN.py to train the RNN model. GPU is highly recommended for reducing the training time.

#  Design materials with desired HOMO-LUMO and internal energy

#  Design molecules active to target proteins

#  MCTS search better molecules
Run python MCTS-RNN.py
