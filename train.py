import os 
import torch 
from tqdm import tqdm
from data_classes.cds_ import HWDataset
import torch.nn as nn
from torch.utils.data import DataLoader

from model_classes.LSTM import LSTMClassifier
#from utils import compute_metrics, evaluate

# def custom_collate_fn(batch, window_size=1000):
#     data_list = []
#     label_list = []
#     for item in batch:
#         data = item['data']
#         label = item['label']
#         num_windows = data.size(0) // window_size
#         for i in range(num_windows):
#             window_data = data[i * window_size:(i + 1) * window_size]
#             data_list.append(window_data)
#             label_list.append(label)
#     return {
#         'label': torch.stack(label_list),
#         'data': torch.stack(data_list)
#     }


if __name__ == '__main__':
    dir = []
    dir.append(os.path.join(os.path.dirname(__file__), 'DATA'))
    dir.append(os.path.join(os.path.dirname(__file__), 'Labels')) 
    train_dataset = HWDataset(dir, 'train')

    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    batch_size = 32


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers = 10,
        #collate_fn=custom_collate_fn  
    )

    for batch in train_dataloader:
        labels = batch['label']
        data = batch['data']
        # devo testare qui la collate function?
        labels = labels.to(device)
        data = data.to(device)
        
        print(labels.shape, data.shape)

    # input_size = 17  #numero di features date dal tensore
    # hidden_size = 64
    # num_layers = 10
    # num_classes = 3
    # model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
    # model.to(device)

    # criterion = torch.nn.CrossEntropyLoss()
    # #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # num_epochs = 100






#PRIMA 
# torch.Size([32]) torch.Size([32, 8192, 17])
# torch.Size([32]) torch.Size([32, 8192, 17])
# torch.Size([23]) torch.Size([23, 8192, 17])

# Numero di etichette torch.Size([32]) pari a 32 una per ciascun campione nel batch.

# torch.Size([32, 57390, 17]) 32: numero di campioni nel batch
#                             57390: numero righe per ciascun campione nella serie temporale
#                             17 feature per ogni row di serie temporale

#DOPO 
# torch.Size([1824]) torch.Size([1824, 1000, 17])
# torch.Size([1824]) torch.Size([1824, 1000, 17])
# torch.Size([1539]) torch.Size([1539, 1000, 17])


# if __name__ == '__main__':
#     dir = []
#     dir.append(os.path.join(os.path.dirname(__file__), 'DATA'))
#     dir.append(os.path.join(os.path.dirname(__file__), 'Labels')) 
    
#     list_sub, max_n_records = chk_csv_data(dir[0])
#     print(len(list_sub))
#     print(max_n_records)
#     print(len(max_n_records))



