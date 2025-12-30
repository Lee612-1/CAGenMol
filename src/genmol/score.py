import os
import glob
import numpy as np
from tdc import Oracle, Evaluator

from rdkit.Chem import MolFromSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from openbabel import pybel

import subprocess
import multiprocessing
from tqdm import tqdm
import random 

def int_div(smiles):
    evaluator = Evaluator(name = 'Diversity')
    return evaluator(smiles)


def get_scores(smiles, receptor_file, box_center, n_process=32):
    smiles_groups = []
    group_size = len(smiles) / n_process
    for i in range(n_process):
        smiles_groups += [smiles[int(i * group_size):int((i + 1) * group_size)]]

    temp_data = []
    pool = multiprocessing.Pool(processes = n_process)
    for index in range(n_process):
        temp_data.append(pool.apply_async(get_scores_subproc, args=(smiles_groups[index], receptor_file, box_center, )))
    pool.close()
    pool.join()
    scores = []
    for index in range(n_process):
        scores += temp_data[index].get()

    for filename in glob.glob("docking/mols/*"):
        if os.path.exists(filename):
            os.remove(filename)

    return scores

def get_scores_subproc(smiles, receptor_file, box_center):
    scores = []
    # mols = [MolFromSmiles(s) for s in smiles]
    oracle_QED = Oracle(name='QED')
    oracle_SA = Oracle(name='SA')

    for i in tqdm(range(len(smiles))):
        if smiles[i]:
            docking_score = docking(smiles[i], receptor_file=receptor_file, box_center=box_center)
            qed_score = oracle_QED(smiles[i])
            sa_score = (10 - oracle_SA(smiles[i])) / 9
            main_score = (-8.18-docking_score)*abs(-8.18-docking_score) + (7/3) * qed_score + (5/6) * sa_score
            scores.append([main_score, docking_score, qed_score, sa_score])
        else:
            scores.append([0.0, 0.0, 0.0, 0.0])

    return scores


def docking(smiles, receptor_file, box_center, box_size=[20, 20, 20]):
    if smiles == "":
        return 99.9
    r = random.randint(0, 256)
    smiles_cleaned = smiles.replace("/", "").replace("\\", "")
    ligand_mol_file = f"/mnt/sdb/liyanting/file_genmol/docking/mols/mol_{smiles_cleaned}_{r}.mol"
    ligand_pdbqt_file = f"/mnt/sdb/liyanting/file_genmol/docking/mols/mol_{smiles_cleaned}_{r}.pdbqt"
    docking_pdbqt_file = f"/mnt/sdb/liyanting/file_genmol/docking/mols/dock_{smiles_cleaned}_{r}.pdbqt"

    # 3D conformation of SMILES
    try:
        run_line = 'obabel -:%s --gen3D -O %s' % (smiles, ligand_mol_file)
        result = subprocess.check_output(run_line.split(), stderr=subprocess.STDOUT,
                    timeout=30, universal_newlines=True)
    except Exception as e:
        # print(e)
        return 99.9

    # docking by quick vina
    try:
        ms = list(pybel.readfile("mol", ligand_mol_file))
        m = ms[0]
        m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        run_line = '/home/liyanting/genmol/qvina02 --receptor %s --ligand %s --out %s' % (receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
        run_line += ' --center_x %s --center_y %s --center_z %s' % (box_center[0], box_center[1], box_center[2])
        run_line += ' --size_x %s --size_y %s --size_z %s' % (box_size[0], box_size[1], box_size[2])
        run_line += ' --cpu %d' % (64)
        run_line += ' --num_modes %d' % (10)
        run_line += ' --exhaustiveness %d ' % (8)
        result = subprocess.check_output(run_line.split(),
                                            stderr=subprocess.STDOUT,
                                            timeout=100, universal_newlines=True)
        result_lines = result.split('\n')
        affinity_list = list()
        check_result = False
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
            affinity_score = affinity_list[0]

    except Exception as e:
        print(e)
        affinity_score = 99.9
    finally:
        for f in [ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass
    return affinity_score


def reverse_sigmoid_transformation(original_score): 
    if original_score > 99:
        return -1.0 
    else: # return (0, 1)
        _low = -12
        _high = -8
        _k = 0.25
        def _reverse_sigmoid_formula(value, low, high, k) -> float:
            try:
                return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
            except:
                return 0

        transformed = _reverse_sigmoid_formula(original_score, _low, _high, _k) 
        return transformed