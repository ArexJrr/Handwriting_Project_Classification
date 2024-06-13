import os 
import torch 
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_classes.cds_ import HWDataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim 
from model_classes.model import Model
from utils import load_config, evaluate
from torch.optim.lr_scheduler import StepLR

"""
Created on Tue May 21 15:46 CET 2024

@author: andreapietro.arena@unikorestudent.it

Some description
"""

if __name__ == '__main__':
    config = load_config(os.path.join(os.path.dirname(__file__), 'config', 'config_RNN.yaml'))
    dir = []
    dir.append(config.data.data_dir)
    dir.append(config.data.label_dir)
    test_dataset = HWDataset(dir, 'test', config.data.pad_tr)
    device = torch.device(config.data.device)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config.training_LSTM.batch_size,
        shuffle=False,
        num_workers= 10) 
    print("")
    
    print("________Tensor[i]________")
    print(f"[i] Padding or Truncate data dim: {config.data.pad_tr}")
    print(f"[i] 1st Tensor info: {(temp := next(iter(test_dataloader))['data'].to(device).shape)} Batch size: {config.training_LSTM.batch_size} Features: {temp[2]}")


    model = Model.LSTM(config.model_LSTM.input_size, config.model_LSTM.hidden_size, config.model_LSTM.output_size, config.model_LSTM.num_layers, config.model_LSTM.dropout, config.model_LSTM.bidirectional)
    model.to(device)
    model.load_state_dict(torch.load("models/{}_model_hw.pt".format(config.training_LST.name)))

    criterion = nn.CrossEntropyLoss()

    test_metrics = evaluate(model, test_dataloader, criterion, device)
    for key, value in test_metrics.items():
        print(f"Test {key}: {value:.4f}")