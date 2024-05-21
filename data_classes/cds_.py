from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np
import os 
import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


class HWDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.data_paths[0] = os.path.join(os.path.dirname(__file__), '..', data_paths[0]) # Datasets
        self.data_paths[1] = os.path.join(os.path.dirname(__file__), '..', data_paths[1]) # Labels 
        
 
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        pass

    def ml_csv_labels(self, file_name):
        df = pd.read_csv(file_name, sep=',')
        return df.iloc[:,[0,-2]]

    def chk_csv_data(self, folder_path):
        empty_subfs = []
        max_n_records = 0 
        folder_path = os.path.join(os.path.dirname(__file__), '..', 'Data')
        for sub_folder in os.listdir(folder_path):
            sub_folder_path = os.path.join(folder_path, sub_folder)
            if os.path.isdir(sub_folder_path):
                files_in_sub_folder = os.listdir(sub_folder_path)
                for file in files_in_sub_folder:
                    file_path = os.path.join(sub_folder_path, file)
                    if os.path.isfile(file_path) and file.endswith('.csv'):
                        df = pd.read_csv(file_path, sep=',' , names=range(21))
                        max_n_records = np.maximum(max_n_records, df.index[-1])
                if not files_in_sub_folder:
                    empty_subfs.append(sub_folder)
        return empty_subfs, max_n_records


if __name__ == '__main__':
    # my_list = ['Data', 'Labels']
    # test = HWDataset(my_list)
    # csv_file = os.path.join(os.path.dirname(__file__), '..', 'Labels', 'Task_1.csv')
    # df = pd.read_csv(csv_file)
    # print(df)
    pass
