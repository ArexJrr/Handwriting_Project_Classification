import os 
import torch 
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_classes.cds_ import HWDataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim 
from model_classes.model import Model
from utils import Utils 
from torch.optim.lr_scheduler import StepLR
#from utils import compute_metrics, evaluate


from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def compute_metrics(predictions, references):

    accuracy = accuracy_score(references, predictions, zero_division=0)
    f1 = f1_score(references, predictions, average='weighted', zero_division=0)
    precision = precision_score(references, predictions, average='weighted', zero_division=0)
    recall = recall_score(references, predictions, average='weighted', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def custom_collate_fn(batch, sub_batch_size=512):
    sub_batches = []
    labels = []
    
    for item in batch:
        data = item['data']
        label = item['label']
        
        num_sub_batches = data.size(0) // sub_batch_size
        
        for i in range(num_sub_batches):
            start_idx = i * sub_batch_size
            end_idx = start_idx + sub_batch_size
            sub_batch = data[start_idx:end_idx]
            sub_batches.append(sub_batch)
            labels.append(label)
    
    data_tensor = torch.stack(sub_batches, dim=0)
    label_tensor = torch.tensor(labels)
    
    return {'data': data_tensor, 'label': label_tensor}



def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    losses = []  

    for batch in tqdm(dataloader, total=len(dataloader), desc="Training..."):
        inputs, labels = batch['data'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        losses.append(loss.item()) 
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validation_step(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Validation..."): #flag quando la uso o per validation o per test
            #inputs, labels = batch['features'].to(device), batch['labels'].to(device)
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







if __name__ == '__main__':
    dir = []
    dir.append(os.path.join(os.path.dirname(__file__), 'DATA'))
    dir.append(os.path.join(os.path.dirname(__file__), 'Labels')) 
    train_dataset = HWDataset(dir, 'train')
    val_dataset = HWDataset(dir, 'test')

    device = torch.device("mps")

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=1,
        num_workers= 10) # se metti più dei core disponibili da erroe
        #collate_fn=custom_collate_fn) 

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=1,
        num_workers= 10) # se metti più dei processori disponibili da erroe
        #collate_fn=custom_collate_fn) 

    input_size, hidden_size, output_size, num_layers, dropout, bidirectional = 17, 512, 3, 8, 0.1, False
    
    model = Model.LSTM(input_size, hidden_size, output_size, num_layers, dropout, bidirectional)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    num_epochs = 10
    for epoch in range(num_epochs):
        avg_loss = train_step(model, train_dataloader, criterion, optimizer, device)
        metrics = validation_step(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {metrics['loss']:.4f}")

        #print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        scheduler.step()













    # predictions = model(inputs)  
    # labels = batch['label']
    # metrics = Utils.compute_metrics(predictions, labels)
    # print(metrics)










    # for batch in train_dataloader:
    #     labels = batch['label']
    #     data = batch['data']
    #     # devo testare qui la collate function?
    #     labels = labels.to(device)
    #     data = data.to(device)
        
    #     print(labels.shape, data.shape)