from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np
import os
import torch 
import warnings
import re
import datetime
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


class HWDataset(Dataset):
    def __init__(self, paths, type_ds):
        self.paths = paths
        self.list_valid_subjects , self.max_len_series = self.chk_csv_data(self.paths[0]) # Ritorna una lista con i soggetti failed più il numero massimo della serie
        self.list_ofsubject = self.select_sub_ds(type_ds) # splitting ds 
        self.data_files = self.data_filecsv_path() # ottenimento path csv 

    def __len__(self):
        return len(self.data_files)  # 91 Soggetti del train 91 soggetti x 21 files 
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        task_id , subject_code = self.identifier_sub_task(self.data_files[idx])    
        label = self.find_label(task_id, subject_code)
        csv_file = self.data_files[idx]
        dataset = pd.read_csv(csv_file, sep=',', names=range(21))
        list_columns_name = dataset.iloc[0]
        dataset = dataset[1:]
        dataset.columns = list_columns_name
        dataset = dataset.iloc[:, :-3].drop(columns=['Sequence'])
        dataset['Phase'] = LabelEncoder().fit_transform(dataset['Phase']) # LabelEncoding 
        dataset['Timestamp'] = dataset['Timestamp'].apply(lambda x: self.convert_timestamp(x))
        dataset = self.pad_and_truncate_data(dataset, 8192)        
        dataset = dataset.astype(float)
        # for  
        item = {
            'label': torch.tensor(label, dtype=torch.long), #perché si ha compatibilità con la crossentropy loss 
            'data': torch.tensor(dataset.values, dtype=torch.float32) 
        }
        return item # creare una lista di mappe con batch32/64 after mapping/truncate?

    def pad_and_truncate_data(self, df, value):
        if len(df) < value:
            padding_rows = value - len(df)
            padding_data = pd.DataFrame({col: [-1] * padding_rows for col in df.columns})
            df_modified = pd.concat([df, padding_data], ignore_index=True)
        elif len(df) > value:
            df_modified = df.iloc[:value]
        else:
            df_modified = df.copy()
        return df_modified

    def convert_timestamp(self, iso_timestamp):
        iso_timestamp = iso_timestamp.rstrip('Z')
        decimal_pos = iso_timestamp.find('.')
        if decimal_pos != -1 and len(iso_timestamp) - decimal_pos - 1 > 6:
            iso_timestamp = iso_timestamp[:decimal_pos + 7]
        dt = datetime.datetime.strptime(iso_timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        timestamp = dt.timestamp()
        return timestamp
    
    def find_label(self, task_id, subject_code):
        csv_template_filename = f'task_{task_id}.csv'
        csv_file_path = os.path.join(self.paths[1], csv_template_filename)
        if os.path.isfile(csv_file_path):
            df = pd.read_csv(csv_file_path, sep=',')
        else: 
            raise Exception("Label file task_{task_id}.csv not found")
        label = int(df[df.iloc[:, 0] == subject_code].iloc[:, [0,-2]].iloc[0,1])
        if label == -1 : raise Exception("Wrong label selected -1 failed task to labelfile")
        return label 

    def identifier_sub_task(self, path): 
        normalized_path = os.path.normpath(path)
        directory, filename = os.path.split(normalized_path)
        subject_code_match = re.search(r'CRC_SUBJECT_(\d+)', directory)
        if subject_code_match:
            subject_code = subject_code_match.group(1)
        else:
            raise Exception("Problem folder name: not found CRC_SUBJECT_000 Number")

        task_id = filename.split('_')[0]

        if subject_code.startswith('0'):
            subject_code = int(str(int(subject_code)))

        else: subject_code = int(subject_code)
        task_id =  re.search(r'Task(\d+)', task_id)
        if task_id:
            task_id = int(task_id.group(1))

        return task_id, subject_code


    def data_filecsv_path(self):
        data_files = []
        for subject_dir in self.list_ofsubject:
            subject_path = os.path.join(self.paths[0], subject_dir)
            if os.path.isdir(subject_path):
                for task_file in os.listdir(subject_path):
                    if task_file.endswith('.csv'):
                        file_path = os.path.join(subject_path, task_file)
                        data_files.append(file_path)
        return data_files

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
        list_sub = []
        max_n_records = [] 
        ds_folder_path = os.path.join(os.path.dirname(__file__), '..', 'Data')
        for sub_folder in os.listdir(ds_folder_path):
            sub_ds_folder_path = os.path.join(ds_folder_path, sub_folder)
            if os.path.isdir(sub_ds_folder_path):
                files_in_sub_folder = os.listdir(sub_ds_folder_path)
                for file in files_in_sub_folder:
                    file_path = os.path.join(sub_ds_folder_path, file)
                    if os.path.isfile(file_path) and file.endswith('.csv'):
                        df = pd.read_csv(file_path, sep=',' , names=range(21))
                        max_n_records.append(df.index[-1])
                if not files_in_sub_folder:
                    empty_subfs.append(sub_ds_folder_path)
                list_sub.append(sub_ds_folder_path)
        list_sub = [sub for sub in list_sub if sub not in empty_subfs]
        if '.DS_Store' in list_sub: list_sub.remove('.DS_Store')
        return list_sub, max_n_records

