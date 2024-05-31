import os 
import torch 
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_classes.cds_ import HWDataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim 
from model_classes.model import Model
from utils import load_config, evaluate, train_step
from torch.optim.lr_scheduler import StepLR
import yaml
from addict import Dict

if __name__ == '__main__':
    config = load_config(os.path.join(os.path.dirname(__file__), 'config', 'config_RNN.yaml'))
    dir = []
    dir.append(config.data.data_dir)
    dir.append(config.data.label_dir)
    train_dataset = HWDataset(dir, 'train', config.data.pad_tr)
    val_dataset = HWDataset(dir, 'val', config.data.pad_tr)

    device = torch.device(config.training_LSTM.device)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.training_LSTM.batch_size,
        shuffle=True,
        num_workers= 10) 

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.training_LSTM.batch_size,
        shuffle=False,
        num_workers= 10) 
    
    print("________Tensor[i]________")
    print(f"[i] Padding or Truncate data dim: {config.data.pad_tr}")
    print(f"[i] 1st Tensor info: {(temp := next(iter(train_dataloader))['data'].to(device).shape)} Batch size: {config.training_LSTM.batch_size} Features: {temp[2]}")




    model = Model.LSTM(config.model_LSTM.input_size, config.model_LSTM.hidden_size, config.model_LSTM.output_size, config.model_LSTM.num_layers, config.model_LSTM.dropout, config.model_LSTM.bidirectional)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training_LSTM.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

    # lambda_lr = lambda epoch: 0.95 ** epoch
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)

    best_accuracy = 0
    best_accuracy_epoch = 0

    for epoch in range(config.training_LSTM.epochs):
        avg_loss = train_step(model, train_dataloader, criterion, optimizer, device)
        metrics = evaluate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{config.training_LSTM.epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {metrics['loss']:.4f}")
        
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_accuracy_epoch = epoch + 1
            torch.save(model.state_dict(), "models/{}_model_hw.py".format(config.training_LST.name))

        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        scheduler.step()