import os
import sys
sys.path.append(os.path.realpath('.'))
from genmol.sampler import Sampler
import pickle
from tqdm import tqdm
import torch


if __name__ == '__main__':
    num_samples = 1000
    task_id = 2
    sampler = Sampler(f'/hpc2ssd/JH_DATA/spooler/yli106/myproject/genmol/ckpt/admet_{task_id}/checkpoints/9.ckpt')#change
    tasks_list = [['hia_hou', 'bbb_martins', 'ames'], 
                  ['lipophilicity_astrazeneca', 'cyp3a4_substrate_carbonmangels', 'dili'],
                  ['solubility_aqsoldb', 'pgp_broccatelli', 'herg']
                 ]
    prop_list = [[1.0, 1.0, 0.0], [5.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    condition = dict()
    condition['input_prop'] = torch.tensor(prop_list[task_id]).unsqueeze(0)
    samples = sampler.conditional_generation(condition, num_samples, softmax_temp=0.5, randomness=0.5)
    print(samples)
    
    with open(f'/hpc2hdd/home/yli106/file1022/res_admet_{task_id}.pkl', 'wb') as f:    
        pickle.dump(samples, f)
