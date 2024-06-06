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
import yaml
from addict import Dict

def info_model(config):
        print("________Model[i]________")
        print(f"[i] Model Name: {config.training.name} ")
        print(f"[i] Epochs: {config.training.epochs}")
        print(f"[i] Input size: {config.model.input_size}")
        print(f"[i] Hidden size: {config.model.hidden_size}")
        print(f"[i] Number of classies: {config.model.output_size}")
        print(f"[i] Number of layers: {config.model.num_layers}")
        print(f"[i] Dropout: {config.model.dropout}")
        ## Aggiungere optmizer e scheduler
        if config.training.name != "RNN" : print(f"[i] Bidirectional state: {config.model.bidirectional}")

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train() #forward pass tengo in considerazione tutti i passaggi
    total_loss = 0.0
    losses = []  

    for batch in tqdm(dataloader, total=len(dataloader), desc="Training..."):
        inputs, labels = batch['data'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() #stacco l'albero di computazione 
        losses.append(loss.item()) 
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_and_evaluate(model, train_dataset, val_dataset, model_config, device, optimizer, scheduler):
    # config.LSTM.
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0
    best_accuracy_epoch = 0
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=model_config.training.batch_size,
        shuffle=True,
        num_workers=model_config.training.num_workers
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=model_config.training.batch_size,
        shuffle=False,
        num_workers=model_config.training.num_workers
    )

    print("________Tensor[i]________")
    print(f"[i] Padding or Truncate data dim: {model_config.model.hidden_size}")
    print(f"[i] 1st Tensor info: {(temp := next(iter(train_dataloader))['data'].to(device).shape)} Batch size: {temp[0]} Features: {temp[2]}")
    

    for epoch in range(model_config.training.epochs):
        avg_loss = train_step(model, train_dataloader, criterion, optimizer, device)
        metrics = evaluate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{model_config.training.epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {metrics['loss']:.4f}")
        
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_accuracy_epoch = epoch + 1
            torch.save(model.state_dict(), f"models/{model_config.training.name}_model_hw.pt")

        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        scheduler.step()


def define_and_run_RNN(config, device, train_dataset, val_dataset):
    model = Model.RNN(config.RNN.model.input_size, config.RNN.model.hidden_size, config.RNN.model.output_size, config.RNN.model.num_layers, config.RNN.model.dropout)
    optimizer = optim.Adam(model.parameters(), lr=config.RNN.training.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    info_model(config.RNN)
    train_and_evaluate(model, train_dataset, val_dataset, config.RNN, device, optimizer, scheduler)

def define_and_run_LSTM(config, device, train_dataset, val_dataset):
    model = Model.LSTM(config.LSTM.model.input_size, config.LSTM.model.hidden_size, config.LSTM.model.output_size, config.LSTM.model.num_layers, config.LSTM.model.dropout, config.LSTM.model.bidirectional)
    optimizer = optim.Adam(model.parameters(), lr=config.LSTM.training.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    info_model(config.LSTM)
    train_and_evaluate(model, train_dataset, val_dataset, config.LSTM, device, optimizer, scheduler)

def define_and_run_GRU(config, device, train_dataset, val_dataset):
    model = Model.GRU(config.GRU.model.input_size, config.GRU.model.hidden_size, config.GRU.model.output_size, config.GRU.model.num_layers, config.GRU.model.dropout, config.GRU.model.bidirectional)
    optimizer = optim.Adam(model.parameters(), lr=config.GRU.training.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    info_model(config.GRU)
    train_and_evaluate(model, train_dataset, val_dataset, config.GRU, device, optimizer, scheduler)


if __name__ == '__main__':
    config = load_config(os.path.join(os.path.dirname(__file__), 'config', 'config_RNN.yaml'))
    dir = [config.data.data_dir, config.data.label_dir]
    train_dataset = HWDataset(dir, 'train', config.data.pad_tr)
    val_dataset = HWDataset(dir, 'val', config.data.pad_tr)
    device = torch.device(config.data.device)
    #define_and_run_RNN(config, device, train_dataset, val_dataset)
    define_and_run_LSTM(config, device, train_dataset, val_dataset)
    #define_and_run_GRU(config, device, train_dataset, val_dataset)











