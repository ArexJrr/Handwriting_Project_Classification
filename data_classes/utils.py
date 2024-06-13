import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np
import tqdm
import yaml
from addict import Dict

"""
Created on Thu Nov 14 12:08:41 CET 2019

@author: firstname.lastname@email.com

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


def compute_metrics(predictions, references, num_classes=3):
        accuracy = accuracy_score(references, predictions)
        f1 = f1_score(references, predictions, average='weighted', zero_division=0)
        precision = precision_score(references, predictions, average='weighted', zero_division=0)
        recall = recall_score(references, predictions, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(references, predictions)
        class_report = classification_report(references, predictions, zero_division=0, output_dict=True)
        
        # try:
        #     roc_auc = roc_auc_score(references, predictions, average='weighted', multi_class='ovr')
        # except ValueError:
        #     roc_auc = np.nan
        
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
            # "roc_auc": roc_auc
}

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Validation..."): #flag quando la uso o per validation o per test
            inputs, labels = batch['data'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            
            # Save predictions and references for computing metrics
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            references.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(predictions, references)
    metrics['loss'] = avg_loss
    return metrics
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(losses, label='Loss per batch')
    # plt.xlabel('Batch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss per Batch')
    # plt.legend()
    # plt.show()






