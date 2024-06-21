import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np
import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from tqdm import tqdm
from addict import Dict

"""
Created on Sat May 25 14:03 CET 2024

@author: andreapietro.arena@unikorestudent.it

Some description
"""

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return Dict(config)
    except FileNotFoundError:
        print(f"Errore: Il file {config_path} non è stato trovato.")
        raise Exception
    except yaml.YAMLError as exc:
        print(f"Errore durante il parsinsg del file YAML: {exc}")

def save_confusion_matrix(conf_matrix, class_names, output_dir, epoch, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    filename = os.path.join(output_dir, f"{model_name}_CM_{epoch+1}.png")
    plt.savefig(filename)
    plt.close()

def save_classification_report(class_report, output_dir, epoch, model_name):
    report_df = pd.DataFrame(class_report).transpose()
    plt.figure(figsize=(12, 8))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f", cbar=True)
    plt.title(f'Classification Report - {model_name} (Epoch {epoch+1})')
    plt.ylabel("Classes")
    plt.xlabel("Metrics")    
    filename = os.path.join(output_dir, f"{model_name}_CR_{epoch+1}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_loss_plot(train_losses, val_losses, output_dir, epoch, model_name):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, epoch+2), y=train_losses, label='Train Loss', color='blue')
    sns.lineplot(x=range(1, epoch+2), y=val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    filename = os.path.join(output_dir, f"{model_name}_{epoch+1}.png")
    plt.savefig(filename)
    plt.close()

def print_confusion_matrix(matrix, class_names):
    print("[i] Confusion Matrix")
    print()
    
    # Print header
    header = "Pred/Actual  " + "  ".join(f"{name:<8}" for name in class_names)
    print(header)
    print("_" * len(header))
    
    # Print rows
    for i, row in enumerate(matrix):
        row_str = f"{class_names[i]:<11}" + "  ".join(f"{value:<8}" for value in row)
        print(row_str)
    print()


def print_classification_report(report):
    print("[i] Classification Report")
    print("{:<15} {:<10} {:<10} {:<10}".format("", "precision", "recall", "f1-score"))
    print("="*45)
    
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            print("{:<15} {:<10.2f} {:<10.2f} {:<10.2f}".format(
                label, 
                metrics.get('precision', 0), 
                metrics.get('recall', 0), 
                metrics.get('f1-score', 0)
            ))
        elif label == "accuracy":
            print("{:<15} {:<10.2f}".format(label, metrics))
    
    # print("_"*45)
    
    # for label in ["macro avg", "weighted avg"]:
    #     metrics = report.get(label, {})
    #     if metrics:
    #         print("{:<15} {:<10.2f} {:<10.2f} {:<10.2f}".format(
    #             label, 
    #             metrics.get('precision', 0), 
    #             metrics.get('recall', 0), 
    #             metrics.get('f1-score', 0)
    #         ))

def compute_metrics(predictions, references, num_classes=3):
        accuracy = accuracy_score(references, predictions)
        f1 = f1_score(references, predictions, average='weighted', zero_division=0)
        precision = precision_score(references, predictions, average='weighted', zero_division=0)
        recall = recall_score(references, predictions, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(references, predictions)
        class_report = classification_report(references, predictions, zero_division=0, output_dict=True)

        print_confusion_matrix(conf_matrix, ['Class 0', 'Class 1', 'Class 2'])
        print_classification_report(class_report)
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report
}

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Validation..."): 
            inputs, labels = batch['data'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()            
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            references.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(predictions, references)
    metrics['loss'] = avg_loss
    return metrics