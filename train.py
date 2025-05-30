from transformers import MambaConfig, MambaModel
from glob import glob
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def token_to_seq(tokens):
    '''
    INPUT:
    tokens: a encodec compressed token with 4 residual layers
    1 5 9... (course layer)
    2 6 10...
    3 7 11...
    4 8 12... (fine layer)
    
    OUTPUT:
    a: a flatten seq
    1 2 3 4 5 6 7 8 9 10 11 12...
    '''
    K, L = tokens.shape
    a = np.zeros((K*L))
    for i in range(K*L):
        a[i] = tokens[i%4, i//4]
    return a

def seq_to_token(seq):
    '''
    INPUT:
    a: a flatten seq
    1 2 3 4 5 6 7 8 9 10 11 12...
    
    OUTPUT:
    tokens: a encodec compressed token with 4 residual layers
    1 5 9... (course layer)
    2 6 10...
    3 7 11...
    4 8 12... (fine layer)
    '''
    L = seq.shape[0]
    print(L)
    a = np.zeros((4, L//4))
    idx = 0
    for i in range(L//4):
        for j in range(4):
            a[j][i] = seq[idx]
            idx+=1
    return a

class Dataset(object):
    def __init__(self, datalist) -> None:
        self.data_path = datalist
    
    def __getitem__(self, idx):
        # shape = [1, 128, slice_length]
        path = self.data_path[idx]
        data = np.load(path, allow_pickle=True)
        a = token_to_seq(data)
        # K, L = data.shape
        # a = np.zeros((K*L))
        # for i in range(K*L):
        #     a[i] = data[i%4, i//4]
        return 	torch.LongTensor(a[0:-4]), torch.LongTensor(a[-4:])
    
    def __len__(self):
        return len(self.data_path)

class Musicmamba(nn.Module):
    def __init__(self):
        super(Musicmamba, self).__init__()
        # parameters setup
        self.card = 2048
        self.dim = 128
        self.token_layer = 4
    
        configuration = MambaConfig(
            vocab_size = self.card,
            hidden_size = self.dim
        )
        self.model = MambaModel(configuration)
        
        self.linear = nn.Linear(self.dim, self.card)
    
    def config(self):
        config = {   
                    'card': self.card,
                    'token_layer': self.token_layer,
                    'dim': self.dim,
                }
        
        return config

    def forward(self, x):
        # B, S = x.shape
        # print(B, S)
        out = self.model(x)
        # print('============== out ==============')
        # print(out)
        # print(out.last_hidden_state.shape)
        logits = self.linear(out.last_hidden_state)
        # print('============== logits ==============')
        # print(logits)
        # print(logits.shape)
        return logits

def temperature_sampling(logits, temperature, topk):
        # probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        logits = torch.Tensor(logits)
        probs = nn.Softmax(dim=0)(logits / temperature)
        probs = np.array(probs)
        if topk == 1:
            prediction = np.argmax(probs)
        else:
            sorted_index = np.argsort(probs)[::-1]
            candi_index = sorted_index[:topk]
            candi_probs = [probs[i] for i in candi_index]
            # normalize probs
            candi_probs /= sum(candi_probs)
            # choose by predicted probs
            prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return prediction

def train():
    EPOCH = 200
    start_epoch = 1
    BATCH = 8
    project_name = 'v2'
    ckpt_folder = '/mnt/gestalt/home/lonian/mamba/model/ckpts/{}'.format(project_name)
    
    device = 'cuda:1'
    
    os.makedirs(ckpt_folder, exist_ok=True)
    
    # dataset
    path = glob('/mnt/gestalt/home/lonian/datasets/mamba_test_token/*.npy')
    train_data = Dataset(path)
    train_loader = DataLoader(dataset=train_data, batch_size = BATCH, shuffle=True)
    
    # model and optimizer
    model = Musicmamba()
    config = model.config()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    with open(os.path.join(ckpt_folder, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    losses_list = []
    for epoch in range(start_epoch, EPOCH+1):
        model.train()
        single_epoch = []
        
        for x, y in tqdm(train_loader, ncols=120):
            output_logit = model(x.to(device))
            y = y.to(device)
            losses = 0
            # print(output_logit.shape) # [B, 6000, 2048]
            output_logit = output_logit.permute(0,2,1)[:, :, -4:]
            # print(output_logit.shape) # [B, 2048, 4]
            for k in range(4):
                loss = nn.CrossEntropyLoss()(output_logit[:, :, k], y[:, k])
                losses += loss
            # print('\n======================================={}=========================================\n'.format(losses))
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            single_epoch.append(losses.to('cpu').mean().item())
            # break
        
        single_epoch = np.array(single_epoch)
        losses_list.append(single_epoch.mean())
        print('>>> Epoch: {}, Loss: {:.5f}'.format(epoch, losses_list[-1]))
            
            
        if epoch % 10 == 0:
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'scheduler': scheduler.state_dict(),
                        'loss': losses_list[-1],
                        }, os.path.join(ckpt_folder, 'epoch_%03d.pkl'%epoch))
            # losses = np.array(losses)
        np.save(os.path.join(ckpt_folder, 'training_loss'), np.array(losses_list))

def test():
    
    os.makedirs('./results', exist_ok=True)
    model_path = '/mnt/gestalt/home/lonian/mamba/model/ckpts/v2/epoch_200.pkl'
    device = 'cuda:0'
    temperature = 1.2
    topk=5
    
    with torch.no_grad():
        # load model
        checkpoint = torch.load(model_path)
        model = Musicmamba().to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        prompt = np.load('/mnt/gestalt/home/lonian/datasets/mamba_test_token/2.npy', allow_pickle=True)
        prompt = prompt[:, :500]
        prompt_seq = token_to_seq(prompt)
        print(prompt_seq.shape)
        
        L = prompt_seq.shape[0]
        # prompt_seq = torch.LongTensor(prompt_seq).to(device)
        while L < 6000:
            print(L, end='\r')
            output_logits = model(torch.LongTensor(np.array([prompt_seq])).to(device))
            _logit = output_logits[0, -4:].to('cpu').detach().numpy()
            for i in range(4):
                word = temperature_sampling(
                        logits=_logit[i], 
                        temperature=temperature,
                        topk=topk)
                prompt_seq = np.concatenate((prompt_seq, np.array([word])))
            L = prompt_seq.shape[0]
    tokens = seq_to_token(prompt_seq)
    np.save('/mnt/gestalt/home/lonian/mamba/model/results/2.npy', tokens)
    pass


def main():
    test()


if __name__ == '__main__':
    main()