import os 
import torch 
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_classes.dataset_management import HW_Dataset_ML, HWDataset_DL
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim 
from model_classes.model import Model_DL
from utils import load_config, evaluate, save_classification_report, save_confusion_matrix
from torch.optim.lr_scheduler import StepLR

"""
Created on Tue May 21 15:46 CET 2024

@author: andreapietro.arena@unikorestudent.it

This is a test class for the Deep Learning (RNN models only) 
"""
def dataloader_fun(test_dataset, config, model_type):
    """
    Summary
    -------
    Creates a DataLoader for the test dataset based on the provided configuration and model type.

    Parameters
    ----------
    test_dataset : Dataset
        The dataset to be loaded for testing.
    config : object
        Configuration object containing settings for the model.
    model_type : str
        The type of model for which the DataLoader is being created.

    Returns
    -------
    test_dataloader : DataLoader
        DataLoader object for the test dataset.
    """
    model_config = getattr(config, model_type)                                  # Retrieve the model configuration for the specified model type.
    batch_size = model_config.training.batch_size                               # Extract batch size from the model configuration.
    if not isinstance(batch_size, int) or batch_size <= 0:                      # Check if batch size is a positive integer.
        raise ValueError(f"Invalid batch_size for {model_type}: {batch_size}")  # Raise an error if batch size is invalid.
    
    test_dataloader = DataLoader(                                               # Create a DataLoader for the test dataset.
        test_dataset,                                                           # Specify the test dataset.
        batch_size=batch_size,                                                  # Set the batch size.
        shuffle=False,                                                          # Do not shuffle the test data.
        num_workers=model_config.training.num_workers                           # Set the number of worker processes for data loading.
    ) 

    print("________Tensor[i]________")
    print(f"[i] Padding or Truncate data dim: {config.data.pad_tr}")
    print(f"[i] 1st Tensor info: {(temp := next(iter(test_dataloader))['data'].to(device).shape)} Batch size: {batch_size} Features: {temp[2]}")

    return test_dataloader

def load_model(model_type, config):
    """
    Summary
    -------
    Loads and initializes a specified model type based on the configuration settings.

    Parameters
    ----------
    model_type : str
        The type of model to load (e.g., 'LSTM', 'GRU').
    config : object
        Configuration object containing model specifications and training details.

    Returns
    -------
    model : nn.Module
        Initialized model instance with loaded weights.
    """
    model_config = getattr(config, model_type)                                  # Get model-specific configuration from the overall config.
    model_class = getattr(Model_DL, model_type)                                 # Get the corresponding model class from the Model_DL module.
    
    model_kwargs = {                                                            # Define model initialization arguments based on configuration settings.
        "input_size": model_config.model.input_size,
        "hidden_size": model_config.model.hidden_size,
        "output_size": model_config.model.output_size,
        "num_layers": model_config.model.num_layers,
        "dropout": model_config.model.dropout
    }
    
    if model_type in ['LSTM', 'GRU']:                                           # Add bidirectional argument if the model type is LSTM or GRU.
        model_kwargs["bidirectional"] = model_config.model.bidirectional
    
    model = model_class(**model_kwargs)                                         # Instantiate the model using the model class and initialized arguments.
    model.load_state_dict(torch.load(f"models/{model_config.training.name}_model_hw.pt"))       # Load pretrained weights into the model from the saved state dictionary file.
    model.to(device)
    return model

def test_and_evaluate_model(model_type, test_dataset, config, class_names, epoch):
    """
    Summary
    -------
    Loads a pretrained model, evaluates its performance on a test dataset, and saves evaluation results.

    Parameters
    ----------
    model_type : str
        The type of model to load (e.g., 'LSTM', 'GRU').
    test_dataset : Dataset
        The dataset used for testing the model.
    config : object
        Configuration object containing model specifications and training details.
    class_names : list
        List of class names for confusion matrix and classification report.
    epoch : int
        Current epoch number for saving results.

    Returns
    -------
    None
    """
    output_dir = f"test_log/DL/{config[model_type].training.name}"               # Define output directory for saving evaluation results
    os.makedirs(output_dir, exist_ok=True)                                       # Create directories if they do not exist
    model = load_model(model_type, config)                                       # Load pretrained model based on model type and configuration
    test_dataloader = dataloader_fun(test_dataset, config, model_type)           # Create DataLoader for the test dataset
    
    criterion = nn.CrossEntropyLoss()                                            
    test_metrics = evaluate(model, test_dataloader, criterion, device)           # Evaluate model performance on the test dataset
    
    conf_matrix = test_metrics["confusion_matrix"]                               # Extract confusion matrix from evaluation metrics   
    class_report = test_metrics["classification_report"]                         # Extract classification report from evaluation metrics
    
    save_confusion_matrix(conf_matrix, class_names, output_dir, epoch, config[model_type].training.name)     # Save confusion matrix and classification report to the output directory
    save_classification_report(class_report, output_dir, epoch, config[model_type].training.name)            # Save confusion matrix and classification report to the output directory
    

if __name__ == '__main__':
    config = load_config(os.path.join(os.path.dirname(__file__), 'config', 'config_RNN.yaml'))
    dir = [config.data.data_dir, config.data.label_dir]
    test_dataset = HWDataset_DL(dir, 'test', config.data.pad_tr)
    device = torch.device(config.data.device)
    test_and_evaluate_model('RNN', test_dataset, config, ["class 0", "class 1", "class 2"], -1)

