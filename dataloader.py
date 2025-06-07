import os
import numpy as np
import torch
import json
import pickle
from utils import *

class Jamendo_Dataset(object):
    '''
    Jamendo dataset
    '''
    def __init__(self, root_path, is_incontext = False, codec_layer = 4, L = 2588) -> None:
        '''
        root_path = /mnt/gestalt/home/lonian/datasets/Jamendo
        '''
        self.root = root_path
        self.audio_token_path = os.path.join(self.root, 'segments_no_vocals_dac')
        # self.meta_path = os.path.join(self.root, 'jamendo_metadata.json')
        self.meta_path = os.path.join(self.root, 'meta_without_SDD_small.json')
        # self.meta_path = os.path.join(self.root, 'electronic_1000.json')
        
        with open(self.meta_path) as f:
            self.meta = json.load(f)
        self.meta_key = list(self.meta.keys())
        
        self.codec_layer = codec_layer
        self.length = L
        self.special_token_id = 1024
        self.is_incontext = is_incontext
    
    def __getitem__(self, idx):
        path = os.path.join(self.audio_token_path, self.meta[self.meta_key[idx]]['path'].replace('.mp3', '.npy'))
        # path = os.path.join(self.audio_token_path, self.meta[idx]['path'].replace('.mp3', '.npy'))
        description = self.meta[self.meta_key[idx]]['rephrased_caption']
        data = np.load(path, allow_pickle=True)
        data = data[:self.codec_layer, :]
        K, L = data.shape
        # print(K, L)
        # data = torch.LongTensor(np.pad(data,((0, 0), (self.length-L, 0)),'constant',constant_values=(0,0)))
        if L >= self.length:
            data = data[:, :self.length]
        else:
            data = torch.LongTensor(np.pad(data,((0, 0), (0, self.length-L)), 'constant', constant_values=(0, self.special_token_id)))
        data = to_delay(data)
        
        mask = (data == self.special_token_id)
        if self.is_incontext:
            return torch.LongTensor(data[:, :-1]), mask[0], torch.LongTensor(data[:, :]), description

        return torch.LongTensor(data[:, :-1]), mask[0], torch.LongTensor(data[:, 1:]), description
    
    def __len__(self):
        return len(self.meta_key)

class MB_Dataset(object):
    '''
    MusicBench dataset
    '''
    def __init__(self, metadata, root_path, codec_layer = 4, L = 500) -> None:
        self.meta = metadata
        self.root = root_path
        self.codec_layer = codec_layer
        self.length = L
        self.number = len(self.meta)
        self.special_token_id = 2048
    
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.meta[idx]['location'][:-4]+'.npy')
        description = self.meta[idx]['main_caption']
        data = np.load(path, allow_pickle=True)
        data = data[:self.codec_layer, :]
        K, L = data.shape
        # print(K, L)
        # data = torch.LongTensor(np.pad(data,((0, 0), (self.length-L, 0)),'constant',constant_values=(0,0)))
        if L >= self.length:
            data = data[:, :self.length]
        else:
            data = torch.LongTensor(np.pad(data,((0, 0), (0, self.length-L)), 'constant', constant_values=(0, self.special_token_id)))
        data = to_delay(data)
        # data = torch.LongTensor(data)
        # mask = torch.ne(torch.LongTensor(data), 2048)
        # return seq[:, 0:499], seq[:, 1:500]
        mask = (data == 2048)
        return torch.LongTensor(data[:, :-1]), mask[0], torch.LongTensor(data[:, 1:]), description
    
    def __len__(self):
        return len(self.meta)

class MIDICaps_Dataset(object):
    '''
    Jamendo dataset
    '''
    def __init__(self, root_path, trv = "train", is_incontext = False, codec_layer = 1, L = 2588) -> None:
        '''
        root_path = /home/yihsin/dataset/midicaps-mini-parsed/trv (train or valid)
        '''
        # get all midi files
        self.root = os.path.join(root_path, trv)
        self.midis = os.listdir(self.root)
        
        # get captions dictionary
        with open(os.path.join(root_path,"captions.pkl"), "rb") as f:
            self.captions = pickle.load(f)
        
        self.codec_layer = codec_layer
        self.length = L
        self.special_token_id = 530 # note: special token id = 530
        self.is_incontext = is_incontext

        print(f"[dataloader] {trv} set initialization done with {len(self.midis)} files.")
    
    def __getitem__(self, idx):
        # -- returns data and description --
        npy_pth = self.midis[idx] # midis are in .npy format
        data = np.load(os.path.join(self.root, npy_pth),allow_pickle=True)[np.newaxis, :]  # length = 5000 maximum
        # data = data[:self.codec_layer, :] 
        description = self.captions[npy_pth]
        
        # -- pad and truncate data --
        K, L = data.shape
        # print(K, L)
        # data = torch.LongTensor(np.pad(data,((0, 0), (self.length-L, 0)),'constant',constant_values=(0,0)))
        if L >= self.length:
            data = data[:, :self.length]
        else:
            data = torch.LongTensor(np.pad(data,((0, 0), (0, self.length-L)), 'constant', constant_values=(0, self.special_token_id)))
        # data = to_delay(data) 
        # will get the same pattern since 
        # 學長好像有說過某個要分成很多個track的delay pattern? 需要去問問看
        data = torch.LongTensor(data)
        
        # -- masking --
        mask = (data == self.special_token_id)
        if self.is_incontext:
            return data[:, :-1], mask[0], data[:, :], description

        # print(type(mask), type(mask[0]))
        return data[:, :-1], mask[0], data[:, 1:], {"description": description}
    
    def __len__(self):
        return len(self.midis)

if __name__ == '__main__':
    # test code pattern
    # train_data = Jamendo_Dataset(root_path = '/mnt/gestalt/home/lonian/datasets/Jamendo')
    # train_loader = DataLoader(dataset=train_data, batch_size = BATCH, shuffle=True, num_workers=4, pin_memory=True)
    train_data = MIDICaps_Dataset(root_path = "/home/yihsin/midicaps-mini-parsed", trv = "train")
    valid_data = MIDICaps_Dataset(root_path = "/home/yihsin/midicaps-mini-parsed", trv = "valid")
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size = 1, shuffle=True, num_workers=4, pin_memory=True)
    
    for batch in train_loader:
        print("x ", batch[0].shape)
        print("mask ", batch[1].shape, type(batch[1]))
        print("y ", batch[2].shape)
        print("caption ", batch[3])
        break  # Remove this to loop over all batches
    
    print(len(train_data))
    print(len(valid_data))
    print('Done')
    
    # test