# from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
from data_classes.dataset_management import HW_Dataset_ML, HWDataset_DL
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim 
from model_classes.model import Model_DL, KNN, SVM
from utils import load_config, evaluate, save_confusion_matrix, save_classification_report, save_loss_plot
from torch.optim.lr_scheduler import StepLR
import yaml
from addict import Dict


"""
Created on Tue May 21 15:46 CET 2024

@author: andreapietro.arena@unikorestudent.it

Some description
"""

# experiment = Experiment(
#     api_key="TJT6yxLyBFHHT2BIEqjzKJjI2",
#     project_name="hw-classification",
#     workspace="arexjrr"
# )

def info_model(config: yaml):
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
    [i] Model Name: RNN 
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


def train_step(model : nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn, optimizer: torch.optim, device: torch.device) -> float:
    """
    Summary
    ----------
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
    model.train()                                                               # Set the model to training mode
    total_loss = 0.0                                                            # Initialize a variable to accumulate the total loss
    losses = []                                                                 # Initialize an empty list to store individual batch losses                                     

    for batch in tqdm(dataloader, total=len(dataloader), desc="Training..."):   # Iterate over batches in the dataloader, showing progress with tqdm
        inputs, labels = batch['data'].to(device), batch['label'].to(device)    # Move batch data and labels to the specified GPU device
        optimizer.zero_grad()                                                   # Reset the gradients accumulated by the optimizer to zero.
        outputs = model(inputs)                                                 # Calculates model predictions for the current batch
        loss = loss_fn(outputs, labels)                                         # Calculates the loss for the current batch
        loss.backward()                                                         # Propagates backward the error (backpropagation)
        optimizer.step()                                                        # Updates the weights of the model
        
        total_loss += loss.item()                                               # Adds the current batch loss to the total (detaches the computation tree)
        losses.append(loss.item())                                              # Saves the current batch loss for possible analysis
    
    avg_loss = total_loss / len(dataloader)                                     # Calculates the average loss over all batches
    return avg_loss


def train_and_evaluate(model: nn.Module, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset, model_config: yaml, device: torch.device, optimizer: torch.optim, scheduler: torch.optim):
    """
    Summary
    -------
    Trains and evaluates a neural network model over multiple epochs using training and validation datasets.

    Parameters
    ----------
    model : nn.Module
        The neural network model to train and evaluate.
    train_dataset : torch.utils.data.Dataset
        Dataset containing training data.
    val_dataset : torch.utils.data.Dataset
        Dataset containing validation data.
    model_config : yaml
        Configuration object containing model training parameters.
    device : torch.device
        Device (CPU or GPU) on which to perform computations.
    optimizer : torch.optim
        Optimization algorithm used to update the model parameters.
    scheduler : torch.optim
        Learning rate scheduler for controlling the learning rate during training.

    Returns
    -------
    None
    """
    model.to(device)                                                                # Move the model to the specified device (CPU or GPU)
    criterion = nn.CrossEntropyLoss()                                               # Define the loss function for classification tasks (CrossEntropyLoss)
    output_dir = []                                                                 # Initialize an empty list for storing output directories                                     
    train_losses = []                                                               # Initialize an empty list to store training losses
    val_losses = []                                                                 # Initialize an empty list to store validation losses
    best_accuracy = 0                                                               # Initialize a variable to track the best validation accuracy
    best_accuracy_epoch = 0                                                         # Initialize a variable to track the epoch with the best validation accuracy
    patience = 5                                                                    # Number of epochs to wait before stopping training if no improvement is observed
    min_delta = 0.001                                                               # Minimum improvement required to consider a change in loss as significant
    best_val_loss = float('inf')                                                    # Initialize the best validation loss to infinity
    patience_counter = 0                                                            # Initialize a counter to track epochs since the last improvement in validation loss

    output_dir.append(f"train_log/{model_config.training.name}/loss_comparison")    # Append directory path for storing loss comparison logs
    output_dir.append(f"train_log/{model_config.training.name}/train_metrics")      # Append directory path for storing training metrics logs
    os.makedirs(output_dir[0], exist_ok=True)                                       # Create directories if they do not exist
    os.makedirs(output_dir[1], exist_ok=True)                                       # Create directories if they do not exist
    
    train_dataloader = DataLoader(                                                  # Create DataLoader for training dataset
        train_dataset, 
        batch_size=model_config.training.batch_size,
        shuffle=True,
        num_workers=model_config.training.num_workers
    )

    val_dataloader = DataLoader(                                                    # Create DataLoader for validation dataset
        val_dataset, 
        batch_size=model_config.training.batch_size,
        shuffle=False,
        num_workers=model_config.training.num_workers
    )

    print(f"[i] 1st Tensor info: {(temp := next(iter(train_dataloader))['data'].to(device).shape)} Batch size: {temp[0]} Features: {temp[2]}")    
    print("________Training[i]________")
    for epoch in range(model_config.training.epochs):                                       # Iterate through epochs for training
        avg_loss = train_step(model, train_dataloader, criterion, optimizer, device)        # Perform one training step and compute average loss
        metrics = evaluate(model, val_dataloader, criterion, device)                        # Evaluate model on validation dataset and compute validation loss and metrics
        val_loss = metrics['loss']

        train_losses.append(avg_loss)                                                        # Append training loss to list
        val_losses.append(val_loss)                                                          # Append val loss to  list
        print()
        print(f"[i] Epoch N: {epoch+1}/{model_config.training.epochs}")                      # Print epoch information and metrics
        print(f"[i] Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"[i] Accuracy on Validation set: {metrics['accuracy']:.2f}")

        #experiment.log_metric("train_loss", avg_loss, epoch=epoch+1)                        # Log metrics to an experiment tracker (e.g., Comet.ml)
        #experiment.log_metric("val_loss", val_loss, epoch=epoch+1)
        #experiment.log_metric("val_accuracy", metrics['accuracy'], epoch=epoch+1)

        if val_loss < best_val_loss - min_delta:                                             # Save the model if the current validation loss is better than the previous best loss
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"models/{model_config.training.name}_model_hw.pt")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:                                                     # Check if early stopping criteria (patience) is met
            print("Early stopping triggered")
            break

        if metrics['accuracy'] > best_accuracy:                                              # Update best accuracy and best epoch if current accuracy is better
            best_accuracy = metrics['accuracy']
            best_accuracy_epoch = epoch + 1

        conf_matrix = metrics['confusion_matrix']                                            # Extract confusion matrix report
        class_report = metrics['classification_report']                                      # Extract  classification report
        class_names = [0, 1, 2]  
    
        save_confusion_matrix(conf_matrix, class_names, output_dir[1], epoch, model_config.training.name)   # Save confusion matrix, classification report, and loss plot for the current epoch
        save_classification_report(class_report, output_dir[1], epoch, model_config.training.name)          # Save confusion matrix, classification report, and loss plot for the current epoch
        save_loss_plot(train_losses, val_losses, output_dir[0], epoch, model_config.training.name)          # Save confusion matrix, classification report, and loss plot for the current epoch

        scheduler.step()                                                                    # Adjust the learning rate using the scheduler



def define_and_run_RNN(config: yaml, device: torch.device, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset):
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
    model = Model_DL.RNN(config.RNN.model.input_size, config.RNN.model.hidden_size, config.RNN.model.output_size, config.RNN.model.num_layers, config.RNN.model.dropout)
    optimizer = optim.Adam(model.parameters(), lr=config.RNN.training.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    info_model(config.RNN)
    #experiment.set_model_graph(str(model))  # Log the model graph to Comet.ml
    train_and_evaluate(model, train_dataset, val_dataset, config.RNN, device, optimizer, scheduler)

def define_and_run_LSTM(config: yaml, device: torch.device, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset):
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
    model = Model_DL.LSTM(config.LSTM.model.input_size, config.LSTM.model.hidden_size, config.LSTM.model.output_size, config.LSTM.model.num_layers, config.LSTM.model.dropout, config.LSTM.model.bidirectional)  # Istance model LSTM passing iperparameters
    optimizer = optim.Adam(model.parameters(), lr=config.LSTM.training.learning_rate)                                                                                                                            # Choice Adam optmizer
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)                                                                                                                                                        # Choice scheduler StepLR
    info_model(config.LSTM)                                                                                                                                                                                      # Print toscreen the model info    
    #experiment.set_model_graph(str(model))                                                                                                                                                                      # Log the model graph to Comet.ml
    train_and_evaluate(model, train_dataset, val_dataset, config.LSTM, device, optimizer, scheduler)                                                                                                             # Train and evaluate the model

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
    model = Model_DL.GRU(config.GRU.model.input_size, config.GRU.model.hidden_size, config.GRU.model.output_size, config.GRU.model.num_layers, config.GRU.model.dropout, config.GRU.model.bidirectional)    # Istance model GRU passing iperparameters
    optimizer = optim.Adam(model.parameters(), lr=config.GRU.training.learning_rate)                                                                                                                        # Choice Adam optmizer
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)                                                                                                                                                   # Choice scheduler StepLR
    info_model(config.GRU)                                                                                                                                                                                  # Print toscreen the model info                   
    #experiment.set_model_graph(str(model))                                                                                                                                                                 # Log the model graph to Comet.ml
    train_and_evaluate(model, train_dataset, val_dataset, config.GRU, device, optimizer, scheduler)                                                                                                         # Train and evaluate the model


def main_DL(config: yaml, dir: list):
    """
    Main function for Deep Learning (DL) model training and evaluation.

    Parameters
    ----------
    config : object
        Configuration object loaded from a YAML file specifying model and data parameters.
    dir : list
        List containing directories for data and labels.

    Returns
    -------
    None
    """
    train_dataset = HWDataset_DL(dir, 'train', config.data.pad_tr)          # Define training dataset
    val_dataset = HWDataset_DL(dir, 'val', config.data.pad_tr)              # Define validation dataset
    device = torch.device(config.data.device)                               # Define GPU device 
    define_and_run_RNN(config, device, train_dataset, val_dataset)          # Define and run the RNN model (train and evaluate)
    #define_and_run_LSTM(config, device, train_dataset, val_dataset)        # Define and run the LSTM model (train and evaluate)
    #define_and_run_GRU(config, device, train_dataset, val_dataset)         # Define and run the GRU model (train and evaluate)
    #experiment.end()                                                       # end experiment tracking by comet.ml

def main_ML(config: yaml, dir: list):
    """
    Main function for Machine Learning (ML) model training and test.

    Parameters
    ----------
    config : object
        Configuration object loaded from a YAML file specifying model and data parameters.
    dir : list
        List containing directories for data and labels.

    Returns
    -------
    None
    """
    train = HW_Dataset_ML(dir, 'train')                                 # Preprocessing: Load and Compute the dataset: train
    #val = HW_Dataset_ML(dir, 'val')
    test = HW_Dataset_ML(dir, 'test')                                   # Preprocessing: Load and Compute the dataset: test

    train_ds = train.get_ds()                                           # Get the ready train dataset for ML algoritms 
    test_ds = test.get_ds()                                             # Get the ready train dataset for ML algoritms

    X_train, y_train = train_ds.iloc[:, :-1], train_ds.iloc[:, -1]      # Separate data from the labels 
    X_test, y_test = test_ds.iloc[:, :-1], test_ds.iloc[:, -1]          # Separate data from the labels

    best_params_knn = KNN.grid_search(X_train, y_train)                 # Call the staticmethod to compute grid search for the KNN model
    #best_params_svm = SVM.grid_search(X_train, y_train)                # Call the staticmethod to compute grid search for the KNN model


    knn = KNN(**best_params_knn)                                        # Implement the grid search iper-parameter and istantiate the model KNN
    knn.train(X_train, y_train)                                         # Train the KNN model
    knn.evaluate(X_test, y_test, 'KNN')                                 # Eval the KNN model

    ##svm = SVM(**best_params_svm)                                      # Implement the grid search iper-parameter and istantiate the model SVM
    # svm = SVM()
    # svm.train(StandardScaler().fit_transform(X_train), y_train)       # Train the KNN model
    # svm.evaluate(StandardScaler().fit_transform(X_test), y_test, 'SVM')   # Eval the KNN model

    
if __name__ == '__main__':
    config = load_config(os.path.join(os.path.dirname(__file__), 'config', 'config_RNN.yaml'))
    dir = [config.data.data_dir, config.data.label_dir]
    main_DL(config, dir)
    #main_ML(config, dir)
















