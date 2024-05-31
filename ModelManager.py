
import numpy as np
import pandas as pd
import torch 
import tqdm
from utils import Utils 
class ModelManager:
    def __init__(self, model, dataloader, loss_fn, optimizer, device, mode, scheduler=None):
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.mode = mode
        if self.mode == 'train': self.flag = True 
        else: self.flag = False

    def set_evaluate_mode(self, dataloader, type_nograd):
        self.dataloader = dataloader
        self.flag = False
        self.mode = type_nograd

    def set_train_mode(self, dataloader):
        self.dataloader = dataloader
        self.flag = True

    def train_step(self):
        if self.mode != 'train':
            raise RuntimeError("Cannot call train_step if you would validate o test model")
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_dataloader, total=len(self.train_dataloader), desc="Training..."):
            inputs, labels = batch['data'].to(self.device), batch['label'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_dataloader)

        return avg_loss

    def evaluate(self):
        if self.flag == True: raise RuntimeError("Cannot call nograd_step if you not update dataloader using: set_nograd_dataloader")
        self.model.eval()
        total_loss = 0.0
        predictions = []
        references = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader, total=len(self.dataloader), desc=self.mode+"..."):
                inputs, labels = batch['data'].to(self.device), batch['label'].to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

                total_loss += loss.item()

                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                references.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.dataloader)
        metrics = compute_metrics(predictions, references)
        metrics['loss'] = avg_loss
        return metrics