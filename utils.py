import sklearn.metrics
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np
import torch.utils
import torch.utils.data
import tqdm
from sklearn.metrics import confusion_matrix
import sklearn
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import torch.nn as nn
from tqdm import tqdm
from addict import Dict
from torch.utils.data import DataLoader, Dataset


"""
Created on Sat May 25 14:03 CET 2024

@author: andreapietro.arena@unikorestudent.it

"""

def load_config(config_path: yaml):
    """
    Summary
    -------
    Loads configuration from a YAML file and converts it to a dictionary.

    Description
    -----------
    This function attempts to open a YAML file specified by the `config_path`, 
    read its contents, and convert it into a `Dict` object. If the file is not found 
    or there is an error during parsing, an exception is raised.

    Parameters
    ----------
    config_path : str
        The path to the YAML configuration file.

    Returns
    -------
    Dict
        The loaded configuration as a `Dict` object.

    Raises
    ------
    Exception
        If the configuration file is not found or if there is an error during YAML parsing.
    """
    try:
        with open(config_path, 'r') as file:                            # Attempt to open the specified file in read mode
            config = yaml.safe_load(file)                               # Load the YAML file content into the config dictionary variable
        return Dict(config)                                             # Return the config dictionary wrapped in a Dict object for easier access
    except FileNotFoundError:
        print(f"Error: The file {config_path} was not found.")          # Handle the case where the specified file is not found
        raise Exception                                                 # Raise a generic exception to indicate a critical error
    except yaml.YAMLError as exc:
        print(f"Error parsing the YAML file: {exc}")                    # Handle YAML parsing errors, if any occur


def save_confusion_matrix(conf_matrix: sklearn.metrics.confusion_matrix , class_names: list, output_dir: str , epoch: int, model_name: str):
    """
    Summary
    -------
    Saves the confusion matrix as a heatmap image file.

    Parameters
    ----------
    conf_matrix : sklearn.metrics.confusion_matrix
        The confusion matrix to be visualized.
    class_names : list
        The list of class names for labeling the axes.
    output_dir : str
        The directory where the confusion matrix image will be saved.
    epoch : int
        The current epoch number, used in the filename.
    model_name : str
        The name of the model, used in the filename.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 8))                                                                                         # Create a new figure with dimensions 10x8 inches
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)       # Create a heatmap of the confusion matrix with annotations (numbers), formatted as integers ('d'), using the 'Blues' colormap, and setting class_names as labels for both x and y axes
    plt.xlabel('Predicted Labels')                                                                                      # Label the x-axis as 'Predicted Labels'
    plt.ylabel('True Labels')                                                                                           # Label the y-axis as 'True Labels'
    plt.title('Confusion Matrix')                                                                                       # Add a title to the plot as 'Confusion Matrix'
    filename = os.path.join(output_dir, f"{model_name}_CM_{epoch+1}.png")                                               # Generate the filename for saving the plot as a PNG file, incorporating the model name and epoch number
    plt.savefig(filename)                                                                                               # Save the current figure to the specified filename
    plt.close()                                                                                                         # Close the current figure to free up memory

def save_classification_report(class_report: sklearn.metrics.classification_report , output_dir: str, epoch: int, model_name: str):
    """
    Summary
    -------
    Saves the classification report as a heatmap image file.

    Parameters
    ----------
    class_report : sklearn.metrics.classification_report
        The classification report containing precision, recall, f1-score, etc., to be visualized.
    output_dir : str
        The directory where the classification report image will be saved.
    epoch : int
        The current epoch number, used in the filename.
    model_name : str
        The name of the model, used in the filename.

    Returns
    -------
    None
    """
    report_df = pd.DataFrame(class_report).transpose()                                          # Create a DataFrame from class_report and transpose it
    plt.figure(figsize=(12, 8))                                                                 # Create a new figure with dimensions 12x8 inches
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f", cbar=True)       # Create a heatmap using the DataFrame, annotate it with values formatted to two decimal places, use the "Blues" color map, and add a color bar
    plt.title(f'Classification Report - {model_name} (Epoch {epoch+1})')                        # Add a title to the figure with the model name and epoch number
    plt.ylabel("Classes")                                                                       # Label the y-axis as "Classes"
    plt.xlabel("Metrics")                                                                       # Label the x-axis as "Metrics"
    filename = os.path.join(output_dir, f"{model_name}_CR_{epoch+1}.png")                       # Generate the image file name based on the model name and epoch number
    plt.savefig(filename, bbox_inches='tight')  # Create                                        # Save the figure as a PNG image file with the specified name, ensuring tight margins
    plt.close()                                                                                 # Close the figure to free up memory

def save_loss_plot(train_losses: float, val_losses: float, output_dir: str, epoch: int, model_name: str):
    """
    Summary
    -------
    Saves the training and validation loss plot as an image file.

    Parameters
    ----------
    train_losses : float
        List of training losses recorded at each epoch.
    val_losses : float
        List of validation losses recorded at each epoch.
    output_dir : str
        The directory where the loss plot image will be saved.
    epoch : int
        The current epoch number, used in the filename.
    model_name : str
        The name of the model, used in the filename.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 6))                                                                 # Create a new figure with specified size.
    sns.lineplot(x=range(1, epoch+2), y=train_losses, label='Train Loss', color='blue')         # Plot training losses.
    sns.lineplot(x=range(1, epoch+2), y=val_losses, label='Validation Loss', color='orange')    # Plot validation losses.
    plt.xlabel('Epochs')                                                                        # Label x-axis.
    plt.ylabel('Loss')                                                                          # Label y-axis.
    plt.title('Training and Validation Loss over Epochs')                                       # Add title to the plot.
    plt.legend()                                                                                # Add legend to the plot.
    plt.grid(True)                                                                              # Add grid to the plot.
    filename = os.path.join(output_dir, f"{model_name}_{epoch+1}.png")                          # Generate filename with model name and epoch number.
    plt.savefig(filename)                                                                       # Save the plot to the specified file.
    plt.close()                                                                                 # Close the plot to free up memory.

def print_confusion_matrix(matrix: sklearn.metrics.confusion_matrix, class_names: list):
    """
    Summary
    -------
    Prints the confusion matrix in a readable format.

    Parameters
    ----------
    matrix : sklearn.metrics.confusion_matrix
        The confusion matrix to be printed. Each row represents the true class,
        and each column represents the predicted class.
    class_names : list of str
        List of class names corresponding to the rows and columns of the confusion matrix.

    Returns
    -------
    None
    """
    print()
    print("[i] Confusion Matrix")    
    header = "Pred/Actual  " + "  ".join(f"{name:<8}" for name in class_names)              # Create header row with class names
    print(header)
    print("_" * len(header))                                                                # Just a style ;)   
    for i, row in enumerate(matrix):                                                        # Iterate through each row in the confusion matrix
        row_str = f"{class_names[i]:<11}" + "  ".join(f"{value:<8}" for value in row)       # Format each row to align with class names and values
        print(row_str)                                                                      # Print formatted row
    print()


def print_classification_report(report: sklearn.metrics.classification_report):
    """
    Summary
    -------
    Prints a classification report in a formatted table.

    Parameters
    ----------
    report : sklearn.metrics.classification_report or dict 
        A object containing classification metrics. Keys are class labels or "accuracy",
        and values are either dictionaries with 'precision', 'recall', and 'f1-score' keys,
        or single values for 'accuracy'.

    Returns
    -------
    None
    """
    print("[i] Classification Report")
    print("{:<15} {:<10} {:<10} {:<10}".format("", "precision", "recall", "f1-score"))     # Print table headers for precision, recall, and f1-score
    print("="*45)
    
    for label, metrics in report.items():                                                  # Iterate through each class label or 'accuracy' in the report
        if isinstance(metrics, dict):                                                      # Check if metrics is a dictionary (class-specific metrics)
            print("{:<15} {:<10.2f} {:<10.2f} {:<10.2f}".format(                           # Print class label and corresponding precision, recall, and f1-score
                label, 
                metrics.get('precision', 0), 
                metrics.get('recall', 0), 
                metrics.get('f1-score', 0)
            ))
        elif label == "accuracy":                                                         # Check if label is 'accuracy' (overall accuracy)
            print("{:<15} {:<10.2f}".format(label, metrics))


def compute_metrics(predictions: np.array, references: np.array, num_classes=3) -> dict:
        """
        Summary
        -------
        Computes and prints various classification metrics based on predicted and reference labels.

        Parameters
        ----------
        predictions : np.array
            Array of predicted class labels.
        references : np.array
            Array of true class labels.
        num_classes : int, optional
            Number of classes in the classification task (default is 3).

        Returns
        -------
        dict
            A dictionary containing computed metrics: accuracy, f1-score, precision, recall,
            confusion matrix, and classification report.
        """
        accuracy = accuracy_score(references, predictions)                                                          # Compute accuracy metric
        f1 = f1_score(references, predictions, average='weighted', zero_division=0)                                 # Compute weighted F1-score metric
        precision = precision_score(references, predictions, average='weighted', zero_division=0)                   # Compute weighted precision metric
        recall = recall_score(references, predictions, average='weighted', zero_division=0)                         # Compute weighted recall metric
        conf_matrix = confusion_matrix(references, predictions)                                                     # Compute confusion matrix
        class_report = classification_report(references, predictions, zero_division=0, output_dict=True)            # Generate classification report as a dictionary
        print_confusion_matrix(conf_matrix, ['Class 0', 'Class 1', 'Class 2'])                                      # Print confusion matrix with class labels
        print_classification_report(class_report)                                                                   # Print classification report to screen
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report
}

def evaluate(model: nn, dataloader: torch.utils.data.DataLoader , loss_fn: nn, device: torch.device) -> dict:
    """
    Summary
    -------
    Evaluates a neural network model on a validation set.

    Parameters
    ----------
    model : nn.Module
        The neural network model to evaluate.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches of validation data.
    loss_fn : nn.Module
        Loss function used to compute the loss between predictions and true labels.
    device : torch.device
        Device (CPU or GPU) on which to perform computations.

    Returns
    -------
    dict
        A dictionary containing evaluation metrics: accuracy, f1-score, precision, recall,
        confusion matrix, classification report, and average loss.
    """
    model.eval()                                                                        # Set model to evaluation mode
    total_loss = 0.0                                                                    # Initialization total loss
    predictions = []                                                                    # Initialization predictions list
    references = []                                                                     # Initialization true references
    with torch.no_grad():                                                               # No compute weights.
        for batch in tqdm(dataloader, total=len(dataloader), desc="Validation..."):     # Iterate over batches in the dataloader
            inputs, labels = batch['data'].to(device), batch['label'].to(device)
            outputs = model(inputs)                                                     # Forward pass through the model
            loss = loss_fn(outputs, labels)                                             # Compute loss
            total_loss += loss.item()                                                   # Accumulate total loss            
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())              # Store predictions and true labels for metrics computation
            references.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)                                             # Compute average loss
    metrics = compute_metrics(predictions, references)                                  # Compute evaluation metrics using computed predictions and references
    metrics['loss'] = avg_loss                                                          # Add average loss to the metrics dictionary
    return metrics