import numpy as np
import os
import pandas as pd
import re
import torch
import warnings
import yaml
import sys
from abc import ABC
from addict import Dict
from importlib import import_module
from itertools import chain
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
sys.path.append('../..')
from utils import load_config
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


"""
Created on Tue May 21 01:43 CET 2024

@author: andreapietro.arena@unikorestudent.it

Some description
"""

class BaseDataset(ABC):
    """
    Summary
    ----------
    BaseDataset class for handling dataset paths and operations (is an abstract class)
    In this class there are a common set of operations that interesting the two subclasses

    Attributes
    ----------
    paths : list
        List of two paths where the data and label folders are located 
    type_ds : str
        Type of dataset to be created (useful for train, validation, test split).
    list_valid_subjects : list
        List of valid subject folders containing tasks.
    max_len_series : int
        List of values length of the series in the datasets.
    list_ofsubject : list
        List of subjects after splitting data.
    data_files : list
        List of absolute paths to the CSV files for the specified dataset type.

    Methods
    -------
    find_label(task_id, subject_code)
        Finds the label for a given task ID and subject code.
    identifier_sub_task(path)
        Identifies the subject and task from a given path based on the folder structure.
    data_filecsv_path
        Retrieves the absolute paths of all CSV files for each subject in the dataset (after split).
    select_sub_ds
        Is an logic function to split the dataset into train, validation and test split
    chk_csv_data
        Checks the CSV data folders and retrieves valid subject directories and the maximum number of records.


    """
    def __init__(self, paths: list, type_ds: str):
        """
        Summary
        --------
        Initializes an instance of HW_Dataset_ML with paths and dataset type.

        Parameters
        --------
            paths (list): 
                The main path where the data folders for each subject are located.
            type_ds (str): 
                Type of dataset to be created ('train', 'val', 'test') for data splitting.

        Methods
        -------
            config (dict): 
                Loaded configuration from 'config_RNN.yaml' file.
            paths (str): 
                Main path where the data folders for each subject are located.
            type_ds (str): 
                Type of dataset ('train', 'val', 'test') to be created.
            list_valid_subjects (list): 
                List of valid subject folders containing tasks.
            max_len_series (int): 
                Maximum length of time series data among all subjects.
            list_ofsubject (list): 
                List of subjects selected based on the dataset type ('train', 'val', 'test').
            data_files (list):
                List of absolute paths of CSV files for the specified dataset type.

        Raises
        -------
            ValueError: If the sum of train_size, val_size, and test_size in the configuration file is not 100.
        """
        self.config = load_config(os.path.join(os.path.dirname(__file__), '..', 'config', 'config_RNN.yaml'))      # Load the config from yaml file
        self.paths = paths                                                                  # Main path where the data folders for each subject are located.
        self.type_ds = type_ds                                                              # Type of dataset to be created (useful for train, validation, test split).
        print("________ Datasets[i] ________")        #how to check istance                              # Print information about the dataset
        self.list_valid_subjects, self.max_len_series = self.chk_csv_data(self.paths[0])    # Call chk_csv_data function to get all valid subject folder paths (containing tasks) and maximum series length.
        self.list_ofsubject = self.select_sub_ds(self.type_ds)                              # Use select_sub_ds function for splitting the data.
        self.data_files = self.data_filecsv_path()                                          # Call data_filecsv_path function to get the absolute paths of the CSV files for the specified dataset type.

    def find_label(self, task_id: int, subject_code:int ) -> int:
        """
        Summary
        ----------
        Finds the label for a given task ID and subject code.

        Parameters
        ----------
        task_id : int
            The ID of the task for which the label is to be found.
        subject_code : int
            The code of the subject for which the label is to be found.

        Returns
        -------
        int
            The label corresponding to the given task ID and subject code.

        Raises
        ------
        Exception
            If the label file for the task is not found.
        Exception
            If the selected label is -1, indicating a failed task.
        """
        csv_template_filename = f'task_{task_id}.csv'                                       # Define the template for how the filenames are structured.
        csv_file_path = os.path.join(self.paths[1], csv_template_filename)                  # Join to the directory path
        if os.path.isfile(csv_file_path):                                                   # If the path is a CSV file, read the file.
            df = pd.read_csv(csv_file_path, sep=',')
        else:
            raise Exception(f"Label file task_{task_id}.csv not found")                     # Raise an exception if the file is not found.
        label = int(df[df.iloc[:, 0] == subject_code].iloc[:, [0, -2]].iloc[0, 1])          # Select the corresponding row, remove other columns, and convert the single value to an integer.
        if label == -1:
            raise Exception("[X] Wrong label selected -1 failed task to labelfile")         # Check if the task is not failed for additional validation.
        return label
    
    def identifier_sub_task(self, path: str) -> tuple[int, int]:
        """
        Summary
        ----------
        Identifies the subject and task from a given path based on the folder structure.

        Parameters
        ----------
        path : str
            The absolute path to the folder containing the subject and task data.

        Returns
        -------
        tuple
            A tuple containing the task ID (int) and the subject code (int).

        Raises
        ------
        Exception
            If the folder name does not contain a valid subject code.
        """
        normalized_path = os.path.normpath(path)                                            # Normalize the absolute folder path.
        directory, filename = os.path.split(normalized_path)                                # Split the directory and filename.  
        subject_code_match = re.search(r'CRC_SUBJECT_(\d+)', directory)                     # Search for an integer after the template in the directory name.
        if subject_code_match:
            subject_code = subject_code_match.group(1)                                      
        else:
            raise Exception("[X] Problem folder name: not found CRC_SUBJECT_000 Number")    # Raise an exception if the value is not found.

        task_id = filename.split('_')[0]                                                    # Extract the task ID from the filename. 
        if subject_code.startswith('0'):                                                    # Remove leading zero if present and convert to integer.
            subject_code = int(str(int(subject_code)))
        else:
            subject_code = int(subject_code)                                               # Convert to integer.

        task_id = re.search(r'Task(\d+)', task_id)                                         # Search for the task ID in the task identifier.
        if task_id:
            task_id = int(task_id.group(1))                                                # Extract and convert the task ID to an integer.
        return task_id, subject_code                                                       # Return the task ID and subject code.


    def data_filecsv_path(self) -> list:
        """
        Summary
        ----------
        Retrieves the absolute paths of all CSV files for each subject in the dataset (after split).

        Parameters
        ----------
        None

        Returns
        -------
        list
            A list of absolute paths to the CSV files for the specified dataset type.

        Raises
        ------
        None
        """
        data_files = []                                                                                                         # Initialize an empty list
        for subject_dir in tqdm(self.list_ofsubject, desc="[+] Retrieval tasks path for each subject ds " + self.type_ds):      # For each subject in list Retrieval all tasks path
            subject_path = os.path.join(self.paths[0], subject_dir)                                                             # Join to the directory subject path
            if os.path.isdir(subject_path):                                                                                     # Check If the passed string as a directory 
                for task_file in os.listdir(subject_path):                                                                      # For each file in current directory
                    if task_file.endswith('.csv'):                                                                              # Check if the task file has extension like .csv
                        file_path = os.path.join(subject_path, task_file)                                                       # Obtain the file absolute path
                        data_files.append(file_path)                                                                            # Append the file path to the list
        print(f"[i] Total CSV tasks for {self.type_ds} ds: ", len(data_files))                                                  # Print the total number of tasks (was founded)
        return data_files                                                                                                       # Return the final list of tasks                                                                           

    def select_sub_ds(self, type_ds: str) -> list:
        """
        Summary
        ----------
        Selects and splits the dataset into training, validation, and test subsets based on the configuration yaml file.

        Parameters
        ----------
        type_ds : str
            The type of dataset to return: 'train', 'val', or 'test'.

        Returns
        -------
        list
            A list of subjects corresponding to the specified dataset type.

        Raises
        ------
        ValueError
            If the sum of train, validation, and test sizes in the configuration yaml file does not equal 100%.
        """
        subjects = self.list_valid_subjects                                                 # List of valid subjects

        if (self.config.data.train_size +
           self.config.data.val_size +
           self.config.data.test_size) != 100 : raise ValueError("[X] Check train, test, val size in config_file.yaml the sum of these must be 100")
        
        train_size = int(len(subjects) * (self.config.data.train_size/100)) #0.75)      # Calculate the number of subjects for train dataset
        val_size = int(len(subjects) * (self.config.data.val_size/100)) #0.15)          # Calculate the number of subjects for val dataset
        test_size = int(len(subjects) * (self.config.data.test_size/100)) #0.10)        # Calculate the number of subjects for test dataset
        
        remaining_tasks = len(subjects) - (train_size + val_size + test_size)           # Adjust train_size to include any remaining tasks
        train_size += remaining_tasks

        train_list_subjects = subjects[:train_size]                                     # Split the subjects into train, validation, and test lists (remeber the list is alfabetic sorted)
        subjects = subjects[train_size:]                                                # Subtract from the len the number of subject and update the list
        val_list_subjects = subjects[:val_size]                                         # Subtract from the len the number of subject and update the list
        subjects = subjects[val_size:]                                                  # Subtract from the len the number of subject and update the list
        test_list_subjects = subjects                                                   # Subtract from the len the number of subject and update the list

        if type_ds.lower() == 'train':                                                  # Return the list of subjects in base the choice (train, validation, test)
            print(f"[i] Number of valid subject: {len(self.list_valid_subjects)}")
            print(f"[i] Train subjects: {len(train_list_subjects)}  | percentage % {self.config.data.train_size}")
            print(f"[i] Val subjects: {len(val_list_subjects)} | percentage % {self.config.data.val_size} ")
            print(f"[i] Test subjects: {len(test_list_subjects)} | percentage % {self.config.data.test_size} ")
            return train_list_subjects
        if type_ds.lower() == 'val': return val_list_subjects
        else:
            print(f"[i] Number of valid subject: {len(self.list_valid_subjects)}  before Oversampling")
            print(f"[i] Test subjects: {len(test_list_subjects)} | percentage % {self.config.data.test_size}") 
            return test_list_subjects
        

    def chk_csv_data(self, ds_folder_path: str) -> tuple[list, list]:
        """
        Summary
        ----------
        Checks the CSV data folders and retrieves valid subject directories and the maximum number of records.

        Parameters
        ----------
        ds_folder_path : str
            The path to the main dataset folder.

        Returns
        -------
        tuple
            A tuple containing:
            - list : A list of valid subject directories.
            - list : A list of the maximum number of records for each CSV file.
        """       
        empty_subfs = []                                                                                            # Initialize empty list used to store empty folder (subjects not to be considered)
        list_sub = []                                                                                               # Initialize empty list used to store valid subjects
        max_n_records = []                                                                                          # Initialize empty list used to store all lenght csv records (one for each csv)
        ds_folder_path = os.path.join(os.path.dirname(__file__), '..', 'Data')                                      # Join to the directory
        for sub_folder in tqdm(os.listdir(ds_folder_path), desc="[+] Checking CSV data folder " + self.type_ds):    # For each subject/folder are into main folder : Checking CSV into sub-data folder
            sub_ds_folder_path = os.path.join(ds_folder_path, sub_folder)                                           # Join to the directory
            if os.path.isdir(sub_ds_folder_path):                                                                   # If the directory as a directory                   
                files_in_sub_folder = os.listdir(sub_ds_folder_path)                                                # Get all files
                for file in files_in_sub_folder:                                                                    # For each file in sub-folder
                    file_path = os.path.join(sub_ds_folder_path, file)                                              # Join to the directory
                    if os.path.isfile(file_path) and file.endswith('.csv'):                                         # Check if the string refers to the valid directory and the file end with .csv
                        df = pd.read_csv(file_path, sep=',', names=range(21))                                       # Read the .csv
                        max_n_records.append(df.index[-1])                                                          # Append the lenght of the dataset
                if not files_in_sub_folder:                                                                         # Check if the directory is empty
                    empty_subfs.append(sub_ds_folder_path)                                                          # Append the directory to the list of empty directories
                list_sub.append(sub_ds_folder_path)                                                                 # Append the directory to the list 
        list_sub = [sub for sub in list_sub if sub not in empty_subfs]                                              # Remove the empty directories from the main directories
        if '.DS_Store' in list_sub:                                                                                 # F*** MACos folder :)
            list_sub.remove('.DS_Store')            
        list_sub = sorted(list_sub)                                                                                 # Sort the list of path directories for to be have a always the same order
        return list_sub, max_n_records                                                                              # Return the list of valid subject and the list of max_records for each ds

class HWDataset_DL(BaseDataset, torch.utils.data.Dataset):
    """
    Summary
    ----------
    HWDataset_DL is a dataset class that inherits from BaseDataset and torch.utils.data.Dataset.
    It handles data loading, preprocessing, and augmentation for handwritten data stored in CSV files.
    
    The class is designed to balance the dataset by applying Gaussian noise to the minority classes,
    pad or truncate data to ensure consistent length, and return formatted data suitable for training
    deep learning models.

    Attributes
    ----------
    paths : list
        List of paths to the dataset CSV files.
    type_ds : str
        The type of dataset, used for different processing pipelines.
    len_ds : int
        Target length for padding or truncating the dataset.
    data_files : list
        List of data files with preprocessed and augmented data.

    Methods
    -------
    __len__():
        Returns the number of items in the dataset.
    __getitem__(idx):
        Retrieves the item (data and label) at the specified index.
    balancing_task():
        Balances the dataset by oversampling the minority classes.
    apply_gaussian_noise(max_item, len_ID_label, csv_file, label):
        Applies Gaussian noise to balance the dataset for a specific class label.
    pad_and_truncate_data(dataset, target_len):
        Pads or truncates the dataset to match the target length.
    """
    def __init__(self, paths: list, type_ds: str, len_ds: int):
        """
        Summary
        ----------
        Initializes the HWDataset_DL object.

        Parameters
        ----------
        paths : list
            List of paths to the dataset CSV files.
        type_ds : str
            The type of dataset, used for different processing pipelines.
        len_ds : int
            Target length for padding or truncating the dataset.

        """
        super().__init__(paths, type_ds)                                                                           # Recall the dad constructor
        self.len_ds = len_ds                                                                                       # Set the lenght for padding o truncate operation    
        self.balancing_task()                                                                                      # Balancing the task with gaussian noise (generate new ds) among the labels

    def __len__(self):
        return len(self.data_files)                                                                                # The len of the object items is defined to the valid csv file taken from valid subject + new csv generated from gaussian noise 

    def __getitem__(self, idx: int) -> torch.tensor:
        """
        Retrieves the item (data and label) at the specified index.

        Parameters
        ----------
        idx : int or tensor
            The index of the item to retrieve.

        Returns
        -------
        dict
            A dictionary containing the data and label tensors. (in both cases)
        """
        if torch.is_tensor(idx):                                                                                   # Convert idx to list if it's a tensor
            idx = idx.tolist()
        if not isinstance(self.data_files[idx], dict):                                                             # If the passed index element is not a dict
            task_id, subject_code = self.identifier_sub_task(self.data_files[idx])                                 # Identifier task di and the subject code from data files
            label = self.find_label(task_id, subject_code)                                                         # Call find label function to get label from task_id and subject_code        
            csv_file = self.data_files[idx]                                                                        # csv file is the element of the list - selected by index                          
            dataset = pd.read_csv(csv_file, sep=',', names=range(21))                                              # read csv and adapt the index from irregular columns            
            list_columns_name = dataset.iloc[0]                                                                    # Set the 1st row to the list columns
            dataset = dataset[1:]                                                                                  # Remove the first row from dataset data file                
            dataset.columns = list_columns_name                                                                    # List of column names
            dataset = dataset.iloc[:, 1:-3].drop(columns=['TimestampRaw'])                                         # Remove the lastest 3 column and the first one, drop the Timestampraw feature     
            dataset['Phase'] = LabelEncoder().fit_transform(dataset['Phase'])                                      # Using LabelEncoder to trasform the Categorical Feature to Numerical format
            dataset = dataset.astype(float)                                                                        # Convert each column data to float
            dataset = self.pad_and_truncate_data(dataset, self.len_ds)                                             # Appling pad or truncate operation to dataset (number of max row)
            item = {                                                                                               # Defining DICT composed from data and label (tensor format)
                'label': torch.tensor(label, dtype=torch.long),
                'data': torch.tensor(dataset.values, dtype=torch.float32)
            }
            return item
        else:                                                                                                      # If the dataset is already tensor (generated ds from apply gaussian noise)                             
            return self.data_files[idx]

    def balancing_task(self):
        """
        Summary
        ----------
        Balances the dataset by oversampling the minority classes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        zero_label_csv, one_label_csv, two_label_csv = [], [], []                                                  # Initialize empty list used to count for each label the number of elements
        for csv_file in self.data_files:                                                                           # Iterate over the list of csv_files and
            task_id, subject_code = self.identifier_sub_task(csv_file)                                             # Identifier task di and the subject code from data files
            label = self.find_label(task_id, subject_code)                                                         # Find the label from task di and the subject
            if label == 0:                                                                                         # In base of task membership assign the csv to the specific label counter
                zero_label_csv.append(csv_file)
            elif label == 1:
                one_label_csv.append(csv_file)
            elif label == 2:
                two_label_csv.append(csv_file)
        max_item = max(len(zero_label_csv), len(one_label_csv), len(two_label_csv))                               # Find the biggest len from the labels 
        pre_len_zero, pre_len_one, pre_len_two = len(zero_label_csv), len(one_label_csv), len(two_label_csv)      # Compute the len of each label (items)
        print(f"[i] CSV label 0: {len(zero_label_csv)} | difference from the biggest is: {max_item - pre_len_zero}")    # Print some information
        print(f"[i] CSV label 1: {len(one_label_csv)} | difference from the biggest is: {max_item - pre_len_one}")      # Print some information
        print(f"[i] CSV label 2: {len(two_label_csv)} | difference from the biggest is: {max_item - pre_len_two}")      # Print some information
        self.apply_gaussian_noise(max_item, pre_len_zero, zero_label_csv, 0)                                     # Apply gaussian noise passing the max lenght that was founded and the len of the specif label 0
        self.apply_gaussian_noise(max_item, pre_len_one, one_label_csv, 1)                                       # Apply gaussian noise passing the max lenght that was founded and the len of the specif label 1
        self.apply_gaussian_noise(max_item, pre_len_two, two_label_csv, 2)                                       # Apply gaussian noise passing the max lenght that was founded and the len of the specif label 2
        print(f"[i] Oversampling data final dim: {len(self.data_files)} | each one label: {max_item}")           # Print the new len of csv files and each one label has the number of task : max_item

    def apply_gaussian_noise(self, max_item: int, len_ID_label: int, csv_file: list, label: int):
        """
        Summary
        ----------
        Applies Gaussian noise to balance the dataset for a specific class label.

        Parameters
        ----------
        max_item : int
            The maximum length among all label lists.
        len_ID_label : int
            The length of the specific label list.
        csv_file : list
            List of CSV file paths for the specific label.
        label : int
            The class label for which Gaussian noise is applied.

        Returns
        -------
        None
        """
        for i in tqdm(range(max_item - len_ID_label), desc=f"[+] Applying Gaussian noise for class {label}"):    # Iter over the label files list (iter dimension is composed by majority len label class - len of specified class)  
            dataset = pd.read_csv(csv_file[i % len_ID_label], sep=',', names=range(21))                          # Read the csv list file with percentage operation index to return from first file when the last file is just processed
            list_columns_name = dataset.iloc[0]                                                                  # Get the column names from the first ds row
            dataset = dataset[1:]                                                                                # Remove the first row from dataset                         
            dataset.columns = list_columns_name                                                                  # Associate the column names to the column of dataset
            dataset = dataset.iloc[:, 1:-3].drop(columns=['TimestampRaw'])                                       # Remove the lastest 3 column and the first one, drop the Timestampraw feature
            dataset['Phase'] = LabelEncoder().fit_transform(dataset['Phase'])                                    # Using LabelEncoder to trasform the Categorical Feature to Numerical format
            dataset = dataset.astype(float)                                                                      # Convert each column data to float
            cols_to_temp_remove = ['Phase', 'PenId', 'Sequence']                                                 # list columns to remove
            cols_index = {col: dataset.columns.get_loc(col) for col in cols_to_temp_remove}                      # Remap the index
            removed_cols = dataset[cols_to_temp_remove].copy()                                                   
            dataset.drop(columns=cols_to_temp_remove, inplace=True)
            noise = np.random.normal(0, 0.1, size=dataset.shape)                                                 # Apply the gaussian noise with mean 0 and standard deviation 0.1
            dataset = dataset + noise                                                                            # Add the noise to the dataset values                                    
            for col, idx in sorted(cols_index.items(), key=lambda x: x[1]):
                dataset.insert(loc=idx, column=col, value=removed_cols[col])
            dataset = self.pad_and_truncate_data(dataset, self.len_ds)                                           # Apply the padding or truncate operation to the ds
            item = {
                'label': torch.tensor(label, dtype=torch.long),
                'data': torch.tensor(dataset.values, dtype=torch.float32)
            }
            self.data_files.append(item)                                                                         # Append to the list of valid subject the new tensor 

    def pad_and_truncate_data(self, dataset: pd.DataFrame, target_len: int) -> pd.DataFrame:
        """
        Summary
        ----------
        Pads or truncates the dataset to match the target length.

        Parameters
        ----------
        dataset : pandas.DataFrame
            The dataset to be padded or truncated.
        target_len : int
            The target length of the dataset after padding or truncation.

        Returns
        -------
        pandas.DataFrame
            The padded or truncated dataset.
        """
        current_len = dataset.shape[0]
        if current_len >= target_len:                                                                           # if the len of the dataset is major or equals to target return a first half of the dataset (from first row to the target len row)
            return dataset.iloc[:target_len]
        else:
            padding = pd.DataFrame(np.zeros((target_len - current_len, dataset.shape[1])), columns=dataset.columns) # Otherwise generate the padding terms witch -1 value (for each row and column values)
            return pd.concat([dataset, padding], ignore_index=True)                                             # Adding the padding rows to the end of dataset
        
class HW_Dataset_ML(BaseDataset):
    """
    HW_Dataset_ML
    -------------
    This class is designed to create a machine learning dataset from the given CSV files.
    It inherits from the BaseDataset class and processes each CSV file to extract relevant 
    features and compute statistical measures (mean and standard deviation) for use in 
    machine learning models.

    Methods
    -------
    get_ds():
        Processes the dataset and returns a formatted DataFrame suitable for machine learning.
    """
    def __init__(self, paths: list, type_ds: str):
        """
        Summary
        ----------
        Initializes the HW_Dataset_ML class.

        Parameters
        ----------
        paths : list
            List of paths to the CSV files to be processed.
        type_ds : str
            Type of dataset (e.g., train, validation, test).
        
        Initializes the BaseDataset with the provided paths and type_ds.
        """
        super().__init__(paths, type_ds)

    def get_ds(self) -> pd.DataFrame:
        """
        Summary
        ----------
        Retrieves a formatted dataset suitable for machine learning models.

        This method processes each CSV file in `self.data_files`, computes statistics (mean and
        standard deviation) for selected columns, and constructs a new DataFrame `dataset_ML`
        containing mean and standard deviation features along with subject ID, task ID, and label.

        Parameters
        ----------
        None

        Returns
        -------
        pandas.DataFrame
            Formatted dataset for machine learning models.
        """
        first_csv_file = self.data_files[0]                                                     # Open the first CSV file to get some util information
        datasetf = pd.read_csv(first_csv_file, sep=',', names=range(21))                        # Read the csv list file with percentage operation index to return from first file when the last file is just processed
        list_columns_name = datasetf.iloc[0]                                                    # Get the column names from the first ds row
        datasetf.columns = list_columns_name                                                    # Remove the first row from dataset
        list_columns_name = datasetf.iloc[0, 1:-4].drop(['Sequence']).tolist()                  # Drop unuseful coluns data
        headers = ['SubjectID', 'TaskID']                                                       # Define the headers
        for column in list_columns_name:                                                        # For each column in list of columns
            headers.extend([f'mean_{column}', f'std_{column}'])                                 # Create new column for each column from the original dataset the new columns in the new dataset (for each column create mean and standard deviation columns feature)
        headers.append('Label')                                                                 # Last column are the label
        dataset_ML = pd.DataFrame(columns=headers)                                              # Set the new columns name

        for idx in range(len(self.data_files)):                                                 # for index in len of data files
            task_id , subject_code = self.identifier_sub_task(self.data_files[idx])             # Identifier task di and the subject code from data files
            label = self.find_label(task_id, subject_code)                                      # Call find label function to get label from task_id and subject_code
            csv_file = self.data_files[idx]                                                     # csv file is the element of the list - selected by index
            dataset = pd.read_csv(csv_file, sep=',', names=range(21))                           # read csv and adapt the index from irregular columns 
            list_columns_name = dataset.iloc[0]                                                 # Set the 1st row to the list columns
            dataset = dataset[1:]                                                               # Remove the first row from dataset data file 
            dataset.columns = list_columns_name                                                 # Associate the column names to the column of dataset   
            dataset = dataset.iloc[:, 1:-4].drop(columns=['Sequence'])                          # Drop specific columns
            dataset['Phase'] = LabelEncoder().fit_transform(dataset['Phase'])                   # Using LabelEncoder to trasform the Categorical Feature to Numerical format
            dataset = dataset.astype(int)                                                       # Convert dataset to int
            mean_values = dataset.mean()                                                        # Compute mean
            std_values = dataset.std()                                                          # Compute standard deviation
            new_row_values = []                                                                 # Initialize new list values

            new_row_values.append(subject_code)                                                 # Append to the list the subject code
            new_row_values.append(task_id)                                                      # Append to the list the task id
            for column in dataset.columns:                                                      # For each column in dataset.column
                new_row_values.append(mean_values[column])                                      # Append to the list the mean values
                new_row_values.append(std_values[column])                                       # Append to the list the std values

            new_row_values.append(label)                                                        # Finaly append the label
            new_row_df = pd.DataFrame([new_row_values], columns=headers)                        # Create and define the dataframe
            dataset_ML = pd.concat([dataset_ML, new_row_df], ignore_index=True)                 # concatenate the new row
        return dataset_ML                                                                       # Return the dataset 
