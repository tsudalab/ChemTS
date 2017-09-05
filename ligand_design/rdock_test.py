from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess

#----parameter & preparation
def rdock_score(compound):

    input_smiles = str(compound)
    num_docking = 3 # number of docking trials
    score_name = '<SCORE>' # <SCORE> or <SCORE.INTER>

    sdf_file = 'tmp_comp.sdf'
    docking_result_file = 'rdock_out'
    min_score = 10**10

    #----Translation from SMILES to sdf
    fw = Chem.SDWriter(sdf_file)
    m1 = Chem.MolFromSmiles(input_smiles)
    try:
        if m1!= None:
            m = Chem.AddHs(m1)

            cid = AllChem.EmbedMolecule(m)
            #fw.write(m)

            opt = AllChem.UFFOptimizeMolecule(m,maxIters=200)
            print(opt)

            fw.write(m)
            fw.close()

            #----rdock calculation
            cmd = '$RBT_ROOT/build/exe/rbdock -r cavity.prm -p $RBT_ROOT/data/scripts/dock.prm -i ' + sdf_file + ' -o ' + docking_result_file + ' -T 1 -n' + str(num_docking)
            proc = subprocess.call( cmd , shell=True)


            #----find the minimum score of rdock from multiple docking results
            f = open(docking_result_file+'.sd')
            lines = f.readlines()
            f.close()



            line_count = 0
            score_line = -1
            for line in lines:
                v_list = line.split()
                if line_count == score_line:
                    if float(v_list[0]) < min_score:
                        min_score = float(v_list[0])

                if len(v_list) <= 1:
                    line_count += 1
                    continue

                if v_list[1] == score_name:
                    score_line = line_count + 1

                line_count += 1


            print('minimum rdock score', min_score)
    except:
        min_score=10**10



    return min_score
