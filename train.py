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

"""
Created on Tue May 21 15:46 CET 2024

@author: andreapietro.arena@unikorestudent.it

Some description
"""

def info_model(config):
    """
    Print the configuration information of the machine learning model.

    Parameters
    ----------
    config : Content .yaml file.
        Configuration object that contains the training and architecture details of the model.

    Returns
    -------
    None
        The function prints the model information without returning any value.

    Example
    -------
    >>> class Config:
    >>> class Training:
    >>> name = "CNN"
    >>> epochs = 10
    >>> class Model:
    >>> input_size = 224
    >>> hidden_size = 128
    >>> output_size = 10
    >>> num_layers = 5
    >>> dropout = 0.5
    >>> bidirectional = true
    >>> config = config()
    >>> config.training = Config.Training()
    >>> config.model = Config.Model()
    >>> info_model(config)
    ________Model[i]________
    [i] Model Name: CNN 
    [i] Epochs: 10
    [i] Input size: 224
    [i] Hidden size: 128
    [i] Number of classies: 10
    [i] Number of layers: 5
    [i] Dropout: 0.5
    [i] Bidirectional state: True
    """
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
    """
    Performs a training step for the machine learning model on a single epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The deep learning model to be trained.
    dataloader : torch.utils.data.DataLoader
        DataLoader that provides batches of data for training.
    loss_fn : callable
        The loss function used to calculate the error between model predictions and targets.
    optimizer : torch.optim.Optimizer.
        The optimizer used to update the model weights.
    device : torch.device
        The device on which to run the model and data (e.g., 'cuda' or 'cpu').

    Returns
    -------
    float
        The average loss (average loss) calculated over all batches of the epoch.

    Example
    -------
    >>> model = MyModel().to(device)
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> loss_fn = torch.nn.CrossEntropyLoss()
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    >>> avg_loss = train_step(model, dataloader, loss_fn, optimizer, device)
    >>> print(f'Average training loss: {avg_loss}')
    """
    model.train() #forward pass tengo in considerazione tutti i passaggi
    total_loss = 0.0
    losses = []  

    for batch in tqdm(dataloader, total=len(dataloader), desc="Training..."):
        inputs, labels = batch['data'].to(device), batch['label'].to(device)
        optimizer.zero_grad()                                                   # Reset the gradients accumulated by the optimizer to zero.
        outputs = model(inputs)                                                 # Calculates model predictions for the current batch
        loss = loss_fn(outputs, labels)                                         # Calculates the loss for the current batch
        loss.backward()                                                         # Propagates backward the error (backpropagation)
        optimizer.step()                                                        # Updates the weights of the model
        
        total_loss += loss.item()                                               # Adds the current batch loss to the total (detaches the computation tree)
        losses.append(loss.item())                                              # Saves the current batch loss for possible analysis
    
    avg_loss = total_loss / len(dataloader)                                     # Calculates the average loss over all batches
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
    """
    Defines and initiates the training of a recurrent neural network (RNN) based on the provided configuration.

    Parameters
    ----------
    config : object
        Configuration object that contains details of the RNN model, training and optimization.
    device : torch.device
        The device on which to run the model and data (e.g., 'cuda' or 'cpu' or 'mps').
    train_dataset : torch.utils.data.Dataset.
        Dataset used for training the model.
    val_dataset : torch.utils.data.Dataset
        Dataset used for model validation.

    Returns
    -------
    None
        The function performs training and evaluation of the model without returning any value.

    Example
    -------
    >>> class Config:
    >>> class RNN:
    >>> class Model:
    >>> input_size = 10
    >>> hidden_size = 20
    >>> output_size = 5
    >>> num_layers = 2
    >>> dropout = 0.5
    >>> class Training:
    >>> learning_rate = 0.001
    >>> config = config()
    >>> config.RNN = Config.RNN()
    >>> config.RNN.model = Config.RNN.Model()
    >>> config.RNN.training = Config.RNN.Training()
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> train_dataset = MyTrainDataset()
    >>> val_dataset = MyValDataset()
    >>> define_and_run_RNN(config, device, train_dataset, val_dataset)
    """
    model = Model.RNN(config.RNN.model.input_size, config.RNN.model.hidden_size, config.RNN.model.output_size, config.RNN.model.num_layers, config.RNN.model.dropout)
    optimizer = optim.Adam(model.parameters(), lr=config.RNN.training.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    info_model(config.RNN)
    train_and_evaluate(model, train_dataset, val_dataset, config.RNN, device, optimizer, scheduler)

def define_and_run_LSTM(config, device, train_dataset, val_dataset):
    """
    Defines and starts the training of an LSTM neural network based on the provided configuration.

    Parameters
    ----------
    config : object
        Configuration object that contains details of the LSTM model, training and optimization.
    device : torch.device
        The device on which to run the model and data (e.g., 'cuda' or 'mps').
    train_dataset : torch.utils.data.Dataset.
        Dataset used for training the model.
    val_dataset : torch.utils.data.Dataset.
        Dataset used for model validation.

    Returns
    -------
    None
        The function performs training and evaluation of the model without returning any value.

    Example
    -------
    >>> class Config:
    >>> class LSTM:
    >>> class Model:
    >>> input_size = 10
    >>> hidden_size = 20
    >>> output_size = 5
    >>> num_layers = 2
    >>> dropout = 0.5
    >>> bidirectional = True
    >>> class Training:
    >>> learning_rate = 0.001
    >>> config = config()
    >>> config.LSTM = Config.LSTM()
    >>> config.LSTM.model = Config.LSTM.Model()
    >>> config.LSTM.training = Config.LSTM.Training()
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> train_dataset = MyTrainDataset()
    >>> val_dataset = MyValDataset()
    >>> define_and_run_LSTM(config, device, train_dataset, val_dataset)
    """
    model = Model.LSTM(config.LSTM.model.input_size, config.LSTM.model.hidden_size, config.LSTM.model.output_size, config.LSTM.model.num_layers, config.LSTM.model.dropout, config.LSTM.model.bidirectional)
    optimizer = optim.Adam(model.parameters(), lr=config.LSTM.training.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    info_model(config.LSTM)
    train_and_evaluate(model, train_dataset, val_dataset, config.LSTM, device, optimizer, scheduler)

def define_and_run_GRU(config, device, train_dataset, val_dataset):
    """
    Defines and starts the training of a GRU neural network based on the provided configuration.

    Parameters
    ----------
    config : object
        Configuration object that contains details of the GRU model, training and optimization.
    device : torch.device
        The device on which to run the model and data (e.g., 'cuda' or 'cpu').
    train_dataset : torch.utils.data.Dataset.
        Dataset used for training the model.
    val_dataset : torch.utils.data.Dataset
        Dataset used for model validation.

    Returns
    -------
    None
        The function performs training and evaluation of the model without returning any value.

    Example
    -------
    >>> class Config:
    >>> class GRU:
    >>> class Model:
    >>> input_size = 10
    >>> hidden_size = 20
    >>> output_size = 5
    >>> num_layers = 2
    >>> dropout = 0.5
    >>> bidirectional = True
    >>> class Training:
    >>> learning_rate = 0.001
    >>> config = config()
    >>> config.GRU = Config.GRU()
    >>> config.GRU.model = Config.GRU.Model()
    >>> config.GRU.training = Config.GRU.Training()
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> train_dataset = MyTrainDataset()
    >>> val_dataset = MyValDataset()
    >>> define_and_run_GRU(config, device, train_dataset, val_dataset)
    """
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











