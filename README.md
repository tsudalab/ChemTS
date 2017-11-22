# ChemTS
Molecule Design using Monte Carlo Tree Search with Neural Rollout. ChemTS can design novel materials with desired HOMO-LUMO gap and interal energy. Combining with rDock, ChemTS can design molecules active to target proteins.  The ChemTS paper is available at https://arxiv.org/abs/1710.00616 .

#  Requirements 
1. [Python](https://www.anaconda.com/download/)>=2.7 
2. [Keras](https://github.com/fchollet/keras) (version 2.0.5) If you installed the newest version of keras, some errors will show up. Please change it back to keras 2.0.5 by pip install keras=2.0.5. 
3. [rdkit](https://anaconda.org/rdkit/rdkit)
4. [rDock](http://rdock.sourceforge.net/installation/)

#  How to use ChemTS? 
For usage, please refer the following instructions.  Currently, the package hasn't been finished very well... If you want to implement your own simulator, please check add_node_type.py. The package will be finished later. 

#  Train a RNN model for molecule generation
1. cd train_RNN
2. Run python train_RNN.py to train the RNN model. GPU is highly recommended for reducing the training time.

#  Design materials with desired HOMO-LUMO and internal energy (coming soon)

#  Design molecules active to target proteins
1. cd ligand_design
2. Run python mcts_ligand.py 

#  MCTS for logP optimization
There are two versions of chemts for logP optimization. The old version considered all possible smiles symbols adding to the tree. The new version chemts only adds with high predictive probabilities to the tree.
1. cd mcts_logp_improved_version
2. Run python mcts_logp.py
