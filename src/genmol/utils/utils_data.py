# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import torch
import datasets
import torch
import pandas as pd
from safe.tokenizer import SAFETokenizer
from rdkit import RDLogger
from genmol.utils.bracket_safe_converter import safe2bracketsafe
RDLogger.DisableLog('rdApp.*')


ROOT_DIR = os.getcwd()


def get_last_checkpoint(save_dir):
    if os.path.exists(save_dir):
        filenames = os.listdir(save_dir)
        if filenames:
            last_filename = sorted(filenames, key=lambda x: int(x[:-5]))[-1]
            return os.path.join(save_dir, last_filename)
    

def get_tokenizer():
    tk = SAFETokenizer.from_pretrained('/home/liyanting/genmol/safe-gpt').get_pretrained()
    tk.add_tokens(['<', '>'])   # for bracket_safe
    return tk


class Collator:
    def __init__(self, config):
        self.tokenizer = get_tokenizer()
        self.max_length = config.model.max_position_embeddings
        self.use_bracket_safe = config.training.get('use_bracket_safe')
    
    def __call__(self, examples):
        if self.use_bracket_safe:
            for example in examples: example['input'] = safe2bracketsafe(example['input'])

        batch = self.tokenizer([example['input'] for example in examples],
                               return_tensors='pt',
                               padding=True,
                               truncation=True,
                               max_length=self.max_length)
        del batch['token_type_ids']
        return batch


class UserDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.data_dict = torch.load(data_path, weights_only=False)
        self.keys = [x for x in list(self.data_dict.keys()) if not x.startswith('t_')]
        # self.keys = list(self.data_dict.keys())
        self.tokenizer = get_tokenizer()
        self.max_length = config.model.max_position_embeddings
        self.use_bracket_safe = config.training.get('use_bracket_safe')
        self.mode = config.mode
        
    def __len__(self):
        return len(self.keys)
    
    def read_coor_data(self, line):
        words = line.strip().split()
        tokens = []
        for i in range(0, len(words), 3):
            coor = [float(words[i].strip()), float(words[i+1].strip()), float(words[i+2].strip())]
            tokens.extend(coor)   # [L * 3]
    
        return torch.tensor(tokens)
    
    def read_aa_dict(self):
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        aa2index = dict()
        for aa in alphabet:
            aa2index[aa] = len(aa2index)
        return aa2index
    
    def read_aa_data(self, line):
        amino_acid_dict = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                   "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                   "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
        aa2index = self.read_aa_dict()
        words = line.strip().split()
        aas = [amino_acid_dict[word.strip().split("_")[-3].strip()] for word in words]
        tokens = []
        for word in aas:
            tokens.append(aa2index[word])
        aa_id = [int(word.strip().split("_")[-2].strip()) for word in words]
        

        return torch.IntTensor(tokens), torch.IntTensor(aa_id)

    def get_surface_aa_feature(self, pocket_aa):
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

    def __getitem__(self, idx):
        item = dict()
        key = self.keys[idx]
        line = self.data_dict[key]
        if self.mode.split('_')[0] == 'surf':
            coord = self.read_coor_data(line['surface']['vertice'])
            aa, aa_ids = self.read_aa_data(line['surface']['atom'])
            item['aa_len'] = len(aa)
            item['input_aa'] = aa
            item['input_coord'] = coord
            item['aa_res_ids'] = aa_ids
        elif self.mode.split('_')[0] == 'surfprop':
            item['surfprop'] = self.get_surface_aa_feature(line['pocket_aa'])

        if self.mode.split('_')[1] == 'weight':
            item['dist_weights'] = line['dist_weights']
        item['pocket_len'] = len(line['rep'])
        item['input_rep'] = line['rep']
        item['input_safe'] = line['ligand_safe']
        item['prop'] = torch.tensor([1.0 if x >=0.5 else 0.0 for x in line['prop']])
        return item
    
    def collate_features(self, samples):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        values = [s for s in samples]
        size = max(v.size(0) for v in values)

        batch_size = len(values)
        res = values[0].new(batch_size, size).fill_(0.0)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][: len(v)])
        return res

    def pad_rep(self, tensors, lengths):
        # tensors: list of (n_i, dim)
        max_len = lengths.max()

        dim = tensors[0].shape[1]
        B = len(tensors)

        padded = torch.zeros(B, max_len, dim)

        for i, t in enumerate(tensors):
            padded[i, :t.shape[0], :] = t

        return padded, lengths

    def collate_fn(self, examples):
        if self.use_bracket_safe:
            for example in examples: example['input_safe'] = safe2bracketsafe(example['input_safe'])
        collate_dict = dict()
        if self.mode.split('_')[0] == 'surf':
            input_aa = [item['input_aa'] for item in examples]
            input_coord = [item['input_coord'] for item in examples]
            aa_ids  = [item['aa_res_ids'] for item in examples]
            aa_len = torch.tensor([item['aa_len'] for item in examples])
            padded_aa_ids = self.collate_features(aa_ids)
            padded_inout_aa = self.collate_features(input_aa)
            padded_input_coord = self.collate_features(input_coord)
            collate_dict = {
                'input_aa': padded_inout_aa,
                'input_coord': padded_input_coord,
                'aa_len': aa_len,
                'aa_res_ids': padded_aa_ids,
            }
        elif self.mode.split('_')[0] == 'surfprop':
            if self.mode.split('_')[1] == 'weight':
                surfprop = [(item['dist_weights'] @ item['surfprop']) for item in examples]
            elif self.mode.split('_')[1] in ['attn', 'mean']:
                surfprop = [item['surfprop'].mean(dim=0) for item in examples]
            else:
                raise ValueError('Invalid mode for surfprop')
            padded_surfprop = torch.stack([x for x in surfprop], dim=0)
            collate_dict['surfprop'] = padded_surfprop

        pocket_len = torch.tensor([item['pocket_len'] for item in examples])
        collate_dict['pocket_len'] = pocket_len
        
        if self.mode.split('_')[1] == 'weight':
            input_rep = [(item['dist_weights'] @ item['input_rep']) for item in examples]
            padded_input_rep = torch.stack([x for x in input_rep], dim=0)
        elif self.mode.split('_')[1] =='attn':
            input_rep = [item['input_rep'] for item in examples]
            padded_input_rep, _ = self.pad_rep(input_rep, pocket_len)

        elif self.mode.split('_')[1] == 'mean':
            input_rep = [item['input_rep'].mean(dim=0) for item in examples]
            padded_input_rep = torch.stack([x for x in input_rep], dim=0)
        else:
            raise ValueError('Invalid mode for input_rep')
        
        collate_dict['input_rep'] = padded_input_rep

        if self.mode.split('_')[-1] == 'prop':
            input_prop = [item['prop'] for item in examples]
            collate_dict['input_prop'] = torch.stack([x for x in input_prop], dim=0)
 
        safes = [item['input_safe'] for item in examples]
        safes_dict = self.tokenizer(safes,
                               return_tensors='pt',
                               padding=True,
                               truncation=True,
                               max_length=self.max_length)
        collate_dict['input_ids'] = safes_dict['input_ids']
        collate_dict['attention_mask'] = safes_dict['attention_mask']
        
        
        
        # Return the batch as a dictionary
        return collate_dict
    

class UserDataset_Admet(torch.utils.data.Dataset):
    def __init__(self, task_id, config):
        self.df = pd.read_csv('/hpc2hdd/home/yli106/file1022/denovo_samples_200k_done.csv')
        self.tasks_list = [['hia_hou', 'bbb_martins', 'ames'], 
                           ['lipophilicity_astrazeneca', 'cyp3a4_substrate_carbonmangels', 'dili'],
                           ['solubility_aqsoldb', 'pgp_broccatelli', 'herg']
                           ]
        self.task = self.tasks_list[task_id]
        self.tokenizer = get_tokenizer()
        self.max_length = config.model.max_position_embeddings
        self.use_bracket_safe = config.training.get('use_bracket_safe')
        self.mode = config.mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = dict()
        line = self.df.iloc[idx]
        item['input_safe'] = line['safe']
        item['prop'] = torch.tensor([float(line[self.task[0]]), float(line[self.task[1]]), float(line[self.task[2]])])
        return item

    def collate_fn(self, examples):
        if self.use_bracket_safe:
            for example in examples: example['input_safe'] = safe2bracketsafe(example['input_safe'])
        collate_dict = dict()
        input_prop = [item['prop'] for item in examples]
        input_prop = torch.stack([x for x in input_prop], dim=0)
        collate_dict['input_prop'] = input_prop
 
        safes = [item['input_safe'] for item in examples]
        safes_dict = self.tokenizer(safes,
                               return_tensors='pt',
                               padding=True,
                               truncation=True,
                               max_length=self.max_length)
        collate_dict['input_ids'] = safes_dict['input_ids']
        collate_dict['attention_mask'] = safes_dict['attention_mask']
        
        
        
        # Return the batch as a dictionary
        return collate_dict


def get_dataloader(config):
    if config.data == 'safe':
        return torch.utils.data.DataLoader(
            datasets.load_dataset('datamol-io/safe-gpt', streaming=True, split='train'),
            batch_size=config.loader.batch_size,
            collate_fn=Collator(config),
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=False,  # streaming
            persistent_workers=True)
    
    elif config.data.startswith('admet'):
        ds = UserDataset_Admet(int(config.data.split('_')[-1]), config)
        # User-defined dataset
        return torch.utils.data.DataLoader(
            ds,
            batch_size=config.loader.batch_size,
            collate_fn=ds.collate_fn,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=True,
            persistent_workers=True)

    ds = UserDataset(config.data, config)
    # User-defined dataset
    return torch.utils.data.DataLoader(
        ds,
        batch_size=config.loader.batch_size,
        collate_fn=ds.collate_fn,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=True,
        persistent_workers=True)