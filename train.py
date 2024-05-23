import os 
import torch 
from tqdm import tqdm
from data_classes.cds_ import HWDataset
import torch.nn as nn
from torch.utils.data import DataLoader

#from utils import compute_metrics, evaluate

if __name__ == '__main__':
    dir = []
    dir.append(os.path.join(os.path.dirname(__file__), 'DATA'))
    dir.append(os.path.join(os.path.dirname(__file__), 'Labels')) 
    train_dataset = HWDataset(dir, 'train')

    device = torch.device('mps')
    batch_size = 32


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: {
            'label': torch.stack([item['label'] for item in x]),
            'data': torch.stack([item['data'] for item in x])
        }
    )

    for batch in train_dataloader:
        labels = batch['label']
        data = batch['data']        
        labels = labels.to(device)
        data = data.to(device)
        print(labels.shape, data.shape)
