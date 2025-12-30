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
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import hydra
import lightning as L
import omegaconf
from omegaconf import DictConfig
import torch
from genmol.model import GenMol
from genmol.utils.utils_data import get_dataloader, get_last_checkpoint
from genmol.utils.ema import ExponentialMovingAverage
import itertools

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

def load_model_from_path(path: str, config):
    model = GenMol.load_from_checkpoint(path, strict=False)
    model.config = config
    if model.ema:
        model.ema = ExponentialMovingAverage(model.backbone.parameters(), decay=model.config.training.ema)
    return model

def train(config: DictConfig):
    wandb_logger = None
    if config.wandb.name is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            **config.wandb
        )

    config.data = 'admet_2'

    model = load_model_from_path('/hpc2ssd/JH_DATA/spooler/yli106/myproject/genmol/model.ckpt', config=config)
    # model = GenMol(config)
    ckpt_path = get_last_checkpoint(config.callback.dirpath)

    train_dataloader = get_dataloader(config)
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=[hydra.utils.instantiate(config.callback)],
        strategy=hydra.utils.instantiate({
            '_target_': 'lightning.pytorch.strategies.DDPStrategy',
            'find_unused_parameters': False
        }),
        logger=wandb_logger,
        enable_progress_bar=True
    )
    trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(config: DictConfig):
    train(config)

if __name__ == '__main__':
    main() 
