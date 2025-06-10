import os
import json
import math
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
# pytorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# lightning
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
# dataloader and model
from dataloader import *
from pl_model import Text_Mmamba_pl
from transformers import T5EncoderModel, T5Tokenizer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def cal_torch_model_params(model):
    '''
    :param model:
    :return:
    '''
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total_params/1000000, 'total_trainable_params': total_trainable_params/1000000}

def parse_opt():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--device', type=str,
                        help='gpu device.', default='cuda')
    parser.add_argument('--project_name', type=str,
                        help='project_name.', default='new_project')
    parser.add_argument('--precision', type=str,
                        help='precision.', default='bf16')
    # path
    # parser.add_argument('--metadata_path', type=str,
    #                     help='metadata path.', default='/mnt/gestalt/home/lonian/datasets/MusicBench/musicbench_train_simba.json')  
    parser.add_argument('--root_path', type=str,
                        help='dataset root path.', default='/home/yihsin/midicaps-mini-parsed')
    parser.add_argument('--ckpt_save_path', type=str,
                        help='specify the dir where the model ckpt save', default='./ckpts')
        
    # about model
    parser.add_argument('--model_type', type=str, choices=['transformer', 'simba', 'mamba', 'hybrid'], 
                        help='model backbone', default='simba')
    parser.add_argument('--layer_num', type=int,
                        help='layers of model', default=12)
    parser.add_argument('--d_state', type=int,
                        help='state size of mamba', default=512)
    parser.add_argument("-i", "--is_incontext", action="store_true")
    
    # about training
    parser.add_argument('--batch', type=int,
                        help='batch size', default=4)
    parser.add_argument('--accumulation_step', type=int,
                        help='accumulation_step', default=32)
    parser.add_argument('--codec_layer', type=int,
                        help='codec_layer', default=1)
    

    # about continue
    parser.add_argument("-c", "--is_continue", help='continue training or not', action="store_true")
    parser.add_argument('--ckpt', type=str, help='ckpt path', default=None)
    
    args = parser.parse_args()
    return args

opt = parse_opt()
# print(opt)

def main():
    if not opt.is_continue:
        ########################################################################
        # training
        EPOCH = 500
        start_epoch = 1
        BATCH = opt.batch
        project_name = opt.project_name
        max_grad_norm = 1
        device = opt.device
        accumulation_step = opt.accumulation_step
        dataset_type = 'midicaps-mini'
        
        is_pure_mamba=False
        if opt.model_type == 'transformer':
            layers = list(range(0, 24))
        elif opt.model_type == 'hybrid':
            layers = [0, 1, 2, 21, 22, 23]
        else:
            layers = []
            if opt.model_type == 'mamba':
                is_pure_mamba=True
        
        # if opt.is_incontext:
        #     condition_methed = 'in_context'
        # else:
        #     condition_methed = 'cross_attention'
        ################################################
        config = {}
        config['training'] = {
            'name': project_name,
            'dataset': dataset_type, 
            'epoch': EPOCH,
            # 'data_number': len(metadata),
            'batch': BATCH,
            'accumulation_step': accumulation_step,
            'precision': opt.precision,
        }
        config['model'] = {
            'layers':opt.layer_num,
            'vocab_size': 530+1, #1024+1,
            'codec_layer': opt.codec_layer,
            'd_model': 512, #1024,
            'drop_p':0.3,
            'd_state':opt.d_state,
            'num_heads': 8,
            'self_atten_layers': layers,
            "is_incontext": opt.is_incontext,
            'is_pure_mamba': is_pure_mamba,
        }
        config['optimizer'] = {
            'optim_lr': 1e-4,
            'weight_decay':0.02,
            'betas': (0.9, 0.999),
        }
        
        config['scheduler'] = {
            'warmup_duration': 100,
            'T_max': EPOCH * BATCH // accumulation_step
        }
        ########################################################################
        # ckpts folder path
        os.makedirs(opt.ckpt_save_path, exist_ok=True)
        ckpt_folder = os.path.join(opt.ckpt_save_path, project_name)
        os.makedirs(ckpt_folder, exist_ok=True)
        
        model = Text_Mmamba_pl(config)
        trainer_params = {
            "precision": config['training']['precision'], #'bf16-mixed',
            "accumulate_grad_batches": opt.accumulation_step,
            "devices": 1,
            "accelerator": "gpu",
            "max_epochs": EPOCH,  # 1000
            "log_every_n_steps": 1,
            "default_root_dir": ckpt_folder,
            'callbacks': [L.pytorch.callbacks.ModelCheckpoint(every_n_train_steps=500, save_top_k=-1)],
            # "callbacks": [EarlyStopping(monitor="training_epoch_mean", mode="min", divergence_threshold=2.0, check_finite=True, check_on_train_epoch_end=True)]
        }
        # lightning.pytorch.callbacks.ModelCheckpoint
        config['training']['model_size'] = cal_torch_model_params(model)
        
        with open(os.path.join(ckpt_folder, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        
        def model_to_dict(model):
            model_dict = {}
            for name, layer in model.named_children():
                model_dict[name] = layer.__class__.__name__  # 紀錄層的類型名稱
                # 如果該層有子模塊，遞迴記錄
                if list(layer.children()):
                    model_dict[name] = model_to_dict(layer)
            return model_dict
        model_structure = model_to_dict(model)
        with open(os.path.join(ckpt_folder, 'model_structure.json'), "w") as f:
            json.dump(model_structure, f, indent=4)

    else:
        
        model_path = "/home/yihsin/simba-ldm-midi/midicaps-gen/r9ka35jh/checkpoints/epoch=83-step=11000.ckpt"
        config_path = "/home/yihsin/simba-ldm-midi/0530-simple-trial/new_project/config.json"
        print(f"[main] continue training from {model_path}")
        # config_path = os.path.join(opt.ckpt[::-1].split('/', 4)[-1][::-1], 'config.json')
        with open(config_path) as f:
            config = json.load(f)
        
        ckpt_folder = os.path.join(opt.ckpt_save_path, config['training']['name'])
        os.makedirs(ckpt_folder, exist_ok=True)
            
        model = Text_Mmamba_pl.load_from_checkpoint(model_path, config)
        trainer_params = {
            "precision": 'bf16', #config['training']['precision'],
            "accumulate_grad_batches": opt.accumulation_step,
            "devices": 1,
            "accelerator": "gpu",
            "max_epochs": config['training']['epoch'],  # 1000
            "log_every_n_steps": 1,
            "default_root_dir": ckpt_folder,
            'callbacks': [L.pytorch.callbacks.ModelCheckpoint(every_n_train_steps=500, save_top_k=-1)],
        }
    # torch.backends.cuda.enable_flash_sdp(True)
    # train_data = Jamendo_Dataset(root_path = opt.root_path, codec_layer=config['model']['codec_layer'], is_incontext = config['model']['is_incontext'])
    # train_loader = DataLoader(dataset=train_data, batch_size = config['training']['batch'], shuffle=True, num_workers=4, pin_memory=True)
    train_data = MIDICaps_Dataset(
        root_path = opt.root_path, 
        trv = "train",
        codec_layer=config['model']['codec_layer'], 
        is_incontext = config['model']['is_incontext']
    )
    train_loader = DataLoader(dataset=train_data, batch_size = config['training']['batch'], shuffle=True, num_workers=4, pin_memory=True)
    
    # wandb logger
    wandb_logger = WandbLogger(project="midicaps-gen", name="simple-trial-1")

    trainer = L.Trainer(logger=wandb_logger, **trainer_params)
    trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=opt.ckpt)


if __name__ == '__main__':
    main()