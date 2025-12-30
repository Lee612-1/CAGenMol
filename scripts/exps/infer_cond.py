import os
import sys
sys.path.append(os.path.realpath('.'))
from genmol.sampler import Sampler
import pickle
from tqdm import tqdm
import torch

def read_coor_data(line):
        words = line.strip().split()
        tokens = []
        for i in range(0, len(words), 3):
            coor = [float(words[i].strip()), float(words[i+1].strip()), float(words[i+2].strip())]
            tokens.extend(coor)   # [L * 3]
    
        return torch.tensor(tokens)
    
def read_aa_dict():
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    aa2index = dict()
    for aa in alphabet:
        aa2index[aa] = len(aa2index)
    return aa2index

def read_aa_data(line):
    amino_acid_dict = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
    aa2index = read_aa_dict()
    words = line.strip().split()
    aas = [amino_acid_dict[word.strip().split("_")[-3].strip()] for word in words]
    tokens = []
    for word in aas:
        tokens.append(aa2index[word])
    aa_id = [int(word.strip().split("_")[-2].strip()) for word in words]

    return torch.tensor(tokens), torch.tensor(aa_id)

def get_surface_aa_feature(pocket_aa):
        HYDROPATHY = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "W": -0.9, "G": -0.4,
                    "T": -0.7, "S": -0.8, "Y": -1.3, "P": -1.6, "H": -3.2, "N": -3.5, "D": -3.5, "Q": -3.5, "E": -3.5,
                    "K": -3.9, "R": -4.5}  # *
        CHARGE = {**{'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.1}, **{x: 0 for x in 'ABCFGIJLMNOPQSTUVWXYZ'}} # *
        POLARITY = {**{x: 1 for x in 'RNDQEHKSTY'}, **{x: 0 for x in "ACGILMFPWV"}}
        ACCEPTOR = {**{x: 1 for x in 'DENQHSTY'}, **{x: 0 for x in "RKWACGILMFPV"}}
        DONOR = {**{x: 1 for x in 'RKWNQHSTY'}, **{x: 0 for x in "DEACGILMFPV"}}
        PMAP = lambda x: [HYDROPATHY[x] / 5, CHARGE[x], POLARITY[x], ACCEPTOR[x], DONOR[x]]

        aa_features = []
        for aa in pocket_aa:
            aa_features.append(PMAP(aa))
        return torch.Tensor(aa_features) 

if __name__ == '__main__':
    num_samples = 100
    sampler = Sampler('/hpc2ssd/JH_DATA/spooler/yli106/myproject/genmol/ckpt/surfprop_attn/checkpoints/44.ckpt')#change
    with open('/hpc2hdd/home/yli106/file1022/test_set_surf_2.pkl', 'rb') as f:
        test_set = pickle.load(f)
    
    res = []
    for k in tqdm(list(test_set.keys())):
        d = test_set[k]
        condition = dict()
        # coord = read_coor_data(d['surface']['vertice'])
        # aa, aa_ids = read_aa_data(d['surface']['atom'])
        # condition['input_coord'], condition['input_aa'], condition['aa_len'] = coord.unsqueeze(0), aa.unsqueeze(0), torch.tensor([len(aa)]).unsqueeze(0)
        # condition['input_rep'] = (d['dist_weights'] @ d['rep']).unsqueeze(0)
        condition['input_rep'] = d['rep'].unsqueeze(0)
        condition['surfprop'] = get_surface_aa_feature(d['pocket_aa']).mean(dim=0).unsqueeze(0)
        samples = sampler.conditional_generation(condition, num_samples, softmax_temp=0.5, randomness=0.5)
        print(samples)
        res.append({'res':samples, 'protein_filename': d['protein_filename'], 'ligand_center_of_mass': d['ligand_center_of_mass']}) 

    with open('/hpc2hdd/home/yli106/file1022/res_surfprop_attn_44.pkl', 'wb') as f:    
        pickle.dump(res, f)
