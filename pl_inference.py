import sys

mode = sys.argv[1]
print(mode)
# pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from pl_model import Text_Mmamba_pl
# import lightning as L
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# others
from glob import glob
import numpy as np
import os
import json
from tqdm import tqdm
import math
# import argparse
from transformers import T5EncoderModel, T5Tokenizer
# from text_simba import MB_Dataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# torch.multiprocessing.set_start_method('spawn')
from utils import *

def create_logger(logger_file_path, name=None):
    import time
    import logging
    
    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    if name is not None:
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    else:
        log_name = '{}.log'.format(name)
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger()  # 设定日志对象
    logger.setLevel(logging.INFO)  # 设定日志等级

    file_handler = logging.FileHandler(final_log_file)  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  # 设置文件输出格式
    console_handler.setFormatter(formatter)  # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

model_path = sys.argv[1]
folder_name = sys.argv[2]
subfolder_name = sys.argv[3]


config_path = model_path[::-1].split('/', 4)[-1][::-1]+'/config.json'
with open(config_path) as f:
    config = json.load(f)

model = Text_Mmamba_pl.load_from_checkpoint(model_path, config)
model.eval()
model.freeze()
# folder_name = 'musicgen_baseline'
save_path = f'/mnt/gestalt/home/lonian/mamba/exp_results_prompt/{folder_name}/{subfolder_name}/dac_token'
os.makedirs (save_path, exist_ok=True)

logger = create_logger(f'/mnt/gestalt/home/lonian/mamba/exp_results_prompt/{folder_name}/{subfolder_name}')

logger.info(f'Is incontext: {config['model']['is_incontext']}')
logger.info(f'Attention layers: {config['model']['self_atten_layers']}')
logger.info(f'Is pure mamba: {config['model']['is_pure_mamba']}')
logger.info(model_path)


import datasets
datasets_200 = datasets.load_from_disk('/mnt/gestalt/home/lonian/datasets/MusicCaps/eval_sub_200_prompt')
loader = DataLoader(dataset=datasets_200, batch_size = 10)

# model = Text_Mmamba_pl.load_from_checkpoint("/mnt/gestalt/home/lonian/mamba/model/ckpts/dac_1_transformer/lightning_logs/version_3/checkpoints/epoch=61-step=25500.ckpt", config)




L = 2588//3
with torch.autocast(device_type="cuda", dtype=torch.float32):
    with torch.no_grad():
        device = 'cuda'
        for i in tqdm(loader):
            if os.path.isfile(os.path.join(save_path, '{}.npy'.format(i['ytid'][0]))):
                continue
            description = i['caption']
            # print(len(i['ytid']))
            prompt_seq = model(description=description, length=L, g_scale=3)
            # print(prompt_seq.shape)

            for b in range(len(i['ytid'])):
                np.save(os.path.join(save_path, '{}.npy'.format(i['ytid'][b])), prompt_seq[b, :, :L])
