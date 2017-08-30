# ChemTS
Molecule Design using Monte Carlo Tree Search with Neural Rollout. ChemTS can design novel materials with desired HOMO-LUMO gap and interal energy. Combining with rDock, ChemTS can design molecules active to target proteins.

#  Requirements 
1. Python>=2.7 [ViennaRNA Package](https://www.tbi.univie.ac.at/RNA/index.html)
2. Keras library
3. Rdkit
4. rDock


#  Train a RNN model for molecule generation
1. Run python train_RNN.py to train the RNN model. GPU is highly recommended for reducing the training time.

#  Design materials with desired HOMO-LUMO and internal energy
1. Run python qspr_training.py train a qspr model.
2. Run python qspr_mcts.py to search novel materials with desired HOMO-LUMO gap and energy.

#  Design molecules active to target proteins
1. Run python ligand_mcts.py 

#  MCTS for logP optimization
1. Run python MCTS-RNN.py
