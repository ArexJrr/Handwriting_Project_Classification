import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def compute_metrics(predictions, references, num_classes=3):
        accuracy = accuracy_score(references, predictions)
        f1 = f1_score(references, predictions, average='weighted', zero_division=0)
        precision = precision_score(references, predictions, average='weighted', zero_division=0)
        recall = recall_score(references, predictions, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(references, predictions)
        class_report = classification_report(references, predictions, zero_division=0, output_dict=True)
        
        try:
            roc_auc = roc_auc_score(references, predictions, average='weighted', multi_class='ovr')
        except ValueError:
            roc_auc = np.nan
        
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
            "roc_auc": roc_auc
        }