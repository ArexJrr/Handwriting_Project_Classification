from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np
import os
import torch 
import warnings
import re
from tqdm import tqdm
import openpyxl 
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

"""
Created on Tue May 30 13:04 CET 2024

@author: andreapietro.arena@unikorestudent.it

Some description
"""
class HW_SVM_Dataset():
    def __init__(self, paths, type_ds):
        self.paths = paths
        self.type_ds = type_ds
        self.list_valid_subjects , self.max_len_series = self.chk_csv_data(self.paths[0]) # Ritorna una lista con i soggetti failed più il numero massimo della serie
        self.list_ofsubject = self.select_sub_ds(self.type_ds) # splitting ds 
        self.data_files = self.data_filecsv_path() # ottenimento path csv 

    def __len__(self):
        return 1  # 91 Soggetti del train 91 soggetti x 21 files 
    

    def get_ds(self):
        first_csv_file = self.data_files[0]
        datasetf = pd.read_csv(first_csv_file, sep=',', names=range(21))
        list_columns_name = datasetf.iloc[0]
        datasetf.columns = list_columns_name
        list_columns_name = datasetf.iloc[0, 1:-4].drop(['Sequence']).tolist()        
        headers = ['SubjectID', 'TaskID']
        for column in list_columns_name:
            headers.extend([f'mean_{column}', f'std_{column}'])
        headers.append('Label')
        dataset_svm = pd.DataFrame(columns=headers)

        for idx in range(len(self.data_files)):
            task_id , subject_code = self.identifier_sub_task(self.data_files[idx])    
            label = self.find_label(task_id, subject_code)
            csv_file = self.data_files[idx]
            dataset = pd.read_csv(csv_file, sep=',', names=range(21))
            list_columns_name = dataset.iloc[0]
            dataset = dataset[1:]
            dataset.columns = list_columns_name
            dataset = dataset.iloc[:, 1:-4].drop(columns=['Sequence'])
            dataset['Phase'] = LabelEncoder().fit_transform(dataset['Phase']) # media e varianza di label encoder?  
            dataset = dataset.astype(int)
            mean_values = dataset.mean()
            std_values = dataset.std()
            new_row_values = []

            new_row_values.append(subject_code)
            new_row_values.append(task_id)
            # Estrai alternativamente le medie e le deviazioni standard delle features
            for column in dataset.columns:
                new_row_values.append(mean_values[column])  # Aggiungi la media della feature
                new_row_values.append(std_values[column])   # Aggiungi la deviazione standard della feature

            new_row_values.append(label)
            new_row_df = pd.DataFrame([new_row_values], columns=headers)
            dataset_svm = pd.concat([dataset_svm, new_row_df], ignore_index=True)
        return dataset_svm 


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
        for subject_dir in tqdm(self.list_ofsubject, desc="Retrieval tasks path for each subject ds "+self.type_ds):
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
        for sub_folder in tqdm(os.listdir(ds_folder_path), desc="Checking CSV data folder: "+self.type_ds):
            sub_ds_folder_path = os.path.join(ds_folder_path, sub_folder)
            if os.path.isdir(sub_ds_folder_path):
                files_in_sub_folder = os.listdir(sub_ds_folder_path)
                for file in files_in_sub_folder:
                    file_path = os.path.join(sub_ds_folder_path, file)
                    if os.path.isfile(file_path) and file.endswith('.csv'):
                        df = pd.read_csv(file_path, sep=',', names=range(21))
                        max_n_records.append(df.index[-1])
                if not files_in_sub_folder:
                    empty_subfs.append(sub_ds_folder_path)
                list_sub.append(sub_ds_folder_path)
        list_sub = [sub for sub in list_sub if sub not in empty_subfs]
        if '.DS_Store' in list_sub: list_sub.remove('.DS_Store')
        list_sub = sorted(list_sub)
        return list_sub, max_n_records


if __name__ == '__main__':
    dir = []
    dir.append(os.path.join(os.path.dirname(__file__), '..', 'DATA'))
    dir.append(os.path.join(os.path.dirname(__file__), '..', 'Labels'))

    train_ds = HW_SVM_Dataset(dir, "train")
    val_ds = HW_SVM_Dataset(dir, "val")
    test_ds = HW_SVM_Dataset(dir, "test")
    new_train_ds = train_ds.get_ds()
    new_train_ds.to_csv('train_SVM.csv', index=False, sep=",")
    new_val_ds = val_ds.get_ds()
    new_val_ds.to_csv('test_SVM.csv', index=False, sep=",")
    new_test_ds = test_ds.get_ds()
    new_test_ds.to_csv('val_SVM.csv', index=False, sep=",")
