from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np
import os 
import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


class HWDataset(Dataset):
    def __init__(self, data_paths, type_ds):
        self.data_paths = data_paths
        self.data_paths[0] = os.path.join(os.path.dirname(__file__), '..', data_paths[0]) # Path Datasets
        self.data_paths[1] = os.path.join(os.path.dirname(__file__), '..', data_paths[1]) # Path Labels 
        self.empty_subject_tasks , self.max_len_series = self.chk_csv_data(self.data_paths[0]) # Ritorna una lista con i soggetti failed più il numero massimo della serie
        self.list_valid_subjects = self.remove_empty_subject_tasks()
        self.list_ofsubject = self.select_sub_ds(type_ds)

    def __len__(self):
        return len(self.data_paths[0])
    
    def __getitem__(self, index):
        pass
    
    def ml_csv_labels(self, file_name):
        df = pd.read_csv(file_name, sep=',')
        return df.iloc[:,[0,-2]]

    def remove_empty_subject_tasks(self):
            subfolders = [
                  entry for entry in os.listdir(self.data_paths[0])
                    if os.path.isdir(self.data_paths[0])
                 ]
            new_subfolders = [sub for sub in subfolders if sub not in self.empty_subject_tasks]
            if '.DS_Store' in new_subfolders: new_subfolders.remove('.DS_Store')
            return new_subfolders


    def select_sub_ds(self, type_ds):
        subjects = self.list_valid_subjects

        train_size = int(len(subjects) * 0.75)
        val_size = int(len(subjects) * 0.15)
        test_size = int(len(subjects) * 0.10)
        
        remaining_tasks = len(subjects) - (train_size + val_size + test_size)
        train_size += remaining_tasks

        train_list_subjects = subjects[:train_size]
        subjects = subjects[train_size:]
        val_list_subjects = subjects[:val_size]
        subjects = subjects[val_size:]
        test_list_subjects = subjects
        
        if type_ds.lower() == 'train': return train_list_subjects
        if type_ds.lower() == 'val': return val_list_subjects
        else: return test_list_subjects
        

    def chk_csv_data(self, ds_folder_path):
        empty_subfs = []
        max_n_records = 0 
        ds_folder_path = os.path.join(os.path.dirname(__file__), '..', 'Data')
        for sub_folder in os.listdir(ds_folder_path):
            sub_ds_folder_path = os.path.join(ds_folder_path, sub_folder)
            if os.path.isdir(sub_ds_folder_path):
                files_in_sub_folder = os.listdir(sub_ds_folder_path)
                for file in files_in_sub_folder:
                    file_path = os.path.join(sub_ds_folder_path, file)
                    if os.path.isfile(file_path) and file.endswith('.csv'):
                        df = pd.read_csv(file_path, sep=',' , names=range(21))
                        max_n_records = np.maximum(max_n_records, df.index[-1])
                if not files_in_sub_folder:
                    empty_subfs.append(sub_folder)
        return empty_subfs, max_n_records


if __name__ == '__main__':
    dir = []
    dir.append(os.path.join(os.path.dirname(__file__), '..', 'DATA')) # Datasets
    dir.append(os.path.join(os.path.dirname(__file__), '..', 'Labels')) # Labels 

    test = HWDataset(dir, 'TRAIN')
