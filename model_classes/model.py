import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
from abc import ABC, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
sys.path.append('../..')
from utils import save_confusion_matrix, save_classification_report, save_loss_plot, print_confusion_matrix, print_classification_report, compute_metrics

"""
Created on Thu May 23 18:17 CET 2024

@author: andreapietro.arena@unikorestudent.it

Some description
"""

#|----------------------------------------------------------------
#|                           DL MODELS                             
#|----------------------------------------------------------------

class Model_DL(ABC):
    """
    Summary
    -------
    The `Model_DL` class serves as an abstract base class for deep learning models.
    It defines the common interface and operations for all derived deep learning models.

    Nested Classes
    --------------
    RNN
        A standard recurrent neural network (RNN) for sequence processing.

    Methods
    -------
    (Abstract methods should be defined here as needed for derived classes)
    """
    class RNN(nn.Module):
        """
        Summary
        -------
        A standard recurrent neural network (RNN) for sequence processing.

        Parameters
        ----------
        input_size : int
            The size of the input vectors.
        hidden_size : int
            The size of the hidden layers of the RNN.
        output_size : int
            The size of the output vector.
        num_layers : int, optional
            The number of layers in the RNN. The default is 1.
        dropout : float, optional
            The dropout rate to be applied. The default is 0.

        Methods
        -------
        forward(x, is_validation=False)
            Performs the forward pass through the RNN network.
        """
        def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers=1, dropout=0):
            """
            Summary
            -------
            Initializes the RNN model with specified parameters. Define the layers type of DL network

            Parameters
            ----------
            input_size : int
                The size of the input vectors.
            hidden_size : int
                The size of the hidden layers of the RNN.
            output_size : int
                The size of the output vector.
            num_layers : int, optional
                The number of layers in the RNN. The default is 1.
            dropout : float, optional
                The dropout rate to be applied. The default is 0.
            """
            super(Model_DL.RNN, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)   # RNN network if num_layer != the nets version is "stacked"
            self.fc = nn.Linear(hidden_size, output_size)                                               # Fully connected layer 
            self.dropout = nn.Dropout(dropout)
            self.num_layers = num_layers                                                          # Dropout layer
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Performs the forward pass through the RNN network.

            Parameters
            ----------
            x : torch.Tensor
                Input data.
            is_validation : bool, optional
                Indicates whether the network is in validation mode (without dropout). The default is False.

            Returns
            -------
            torch.Tensor
                The output vector of the RNN network.
            """
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_layers, batch_size, self.rnn.hidden_size).to(x.device)           # Initialize hidden state 
            #h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device)                         # Initialize hidden state with zeros.
            out, _ = self.rnn(x, h0)                                                                   # Perform RNN forward pass.
            out = self.dropout(out)                                                                    # Apply droput (if is in validation mode torch.nograd detach this layer)
            out = self.fc(out[:, -1, :])                                                               # Apply the fully connected layer to make prevision
            return out

    class LSTM(nn.Module):
        """
        Summary
        -------
        An LSTM neural network for sequence processing, with optional support for bidirectionality.

        Parameters
        ----------
        input_size : int
            The size of the input vectors.
        hidden_size : int
            The size of the hidden layers of the LSTM.
        output_size : int
            The size of the output vector.
        num_layers : int, optional
            The number of layers in the LSTM. The default is 1.
        dropout : float, optional
            The dropout rate to be applied. The default is 0.
        bidirectional : bool, optional
            Indicates whether the LSTM is bidirectional. The default is False.

        Methods
        -------
        forward(x, is_validation=False)
            Performs the forward pass through the LSTM network.
        """
        def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers=1, dropout=0, bidirectional=False):
            """
            Summary
            -------
            Initializes the LSTM model with specified parameters.

            Parameters
            ----------
            input_size : int
                The size of the input vectors.
            hidden_size : int
                The size of the hidden layers of the LSTM.
            output_size : int
                The size of the output vector.
            num_layers : int, optional
                The number of layers in the LSTM. The default is 1.
            dropout : float, optional
                The dropout rate to be applied. The default is 0.
            bidirectional : bool, optional
                Indicates whether the LSTM is bidirectional. The default is False.
            """
            super(Model_DL.LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)        # Initalize the LSTM model
            self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)                                             # Define the layer fully connected with the particular actention at bidiretional status 
            self.dropout = nn.Dropout(dropout)                                                                                              # Define the dropout layer for normalization

        def forward(self, x:torch.Tensor) -> torch.Tensor:
            """
            Summary
            -------
            Performs the forward pass through the LSTM network.

            Parameters
            ----------
            x : torch.Tensor
                Input data.

            Returns
            -------
            torch.Tensor
                The output vector of the LSTM network.
            """
            num_directions = 2 if self.lstm.bidirectional else 1                                                        # Determine the number of directions (1 for unidirectional, 2 for bidirectional).
            h0 = torch.zeros(self.lstm.num_layers * num_directions, x.size(0), self.lstm.hidden_size).to(x.device)      # Initialize hidden state with zeros.
            c0 = torch.zeros(self.lstm.num_layers * num_directions, x.size(0), self.lstm.hidden_size).to(x.device)      # Initialize cell state with zeros.
            out, _ = self.lstm(x, (h0, c0))                                                                             # Perform LSTM forward pass
            out = self.dropout(out) #if not is_validation else out                                                      # Apply dropout.
            out = self.fc(out[:, -1, :])                                                                                # Apply the fully connected layer on the last time step.
            return out
        
    class GRU(nn.Module):
        """
        Summary
        -------
        A GRU neural network for sequence processing, with optional support for bidirectionality.

        Parameters
        ----------
        input_size : int
            The size of the input vectors.
        hidden_size : int
            The size of the hidden layers of the GRU.
        output_size : int
            The size of the output vector.
        num_layers : int, optional
            The number of layers in the GRU. The default is 1.
        dropout : float, optional
            The dropout rate to be applied. The default is 0.
        bidirectional : bool, optional
            Indicates whether the GRU is bidirectional. The default is False.

        Methods
        -------
        forward(x, is_validation=False)
            Performs the forward pass through the GRU network.
        """
        def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0, bidirectional=False):
            """
            Summary
            -------
            Initializes the GRU model with specified parameters.

            Parameters
            ----------
            input_size : int
                The size of the input vectors.
            hidden_size : int
                The size of the hidden layers of the GRU.
            output_size : int
                The size of the output vector.
            num_layers : int, optional
                The number of layers in the GRU. The default is 1.
            dropout : float, optional
                The dropout rate to be applied. The default is 0.
            bidirectional : bool, optional
                Indicates whether the GRU is bidirectional. The default is False.
            """
            super(Model_DL.GRU, self).__init__()                                                                                    
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)  # Initialize GRU model
            self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)                                     # Define fully connected layer
            self.dropout = nn.Dropout(dropout)                                                                                      # Define normalization layer dropout            
        
        def forward(self, x:torch.Tensor) -> torch.Tensor:
            """
            Summary
            -------
            Performs the forward pass through the GRU network.

            Parameters
            ----------
            x : torch.Tensor
                Input data.

            Returns
            -------
            torch.Tensor
                The output vector of the GRU network.
            """
            num_directions = 2 if self.gru.bidirectional else 1                                                        # Determine the number of directions (1 for unidirectional, 2 for bidirectional).
            h0 = torch.zeros(self.gru.num_layers * num_directions, x.size(0), self.gru.hidden_size).to(x.device)       # Initialize hidden state with zeros.
            out, _ = self.gru(x, h0)                                                                                   # Perform LSTM forward pass
            out = self.dropout(out)                                                                                    # Apply dropout.
            out = self.fc(out[:, -1, :])                                                                               # Apply the fully connected layer on the last time step.
            return out

#|----------------------------------------------------------------
#|                           ML MODELS                             
#|----------------------------------------------------------------
class Model_ML(ABC):
    """
    Summary
    -------
    The `Model_ML` class is an abstract base class (ABC) for machine learning models.
    It defines the common interface and operations for all derived machine learning models.

    Attributes
    ----------
    model : None
        Placeholder for the machine learning model instance.

    Methods
    -------
    train(X_train, y_train)
        Abstract method for training the model. Must be implemented by derived classes.
    predict(X_test)
        Abstract method for predicting labels. Must be implemented by derived classes.
    evaluate(X_test, y_test, name_model=None)
        Evaluates the model using test data and computes metrics. Saves the confusion matrix and classification report.
    """
    def __init__(self):
        """
        Summary
        -------
        Initializes the `Model_ML` class with a placeholder for the machine learning model.
        """
        self.model = None
    
    @abstractmethod
    def train(self, X_train: np.array, y_train: np.array):
        """
        Summary
        -------
        Abstract method for training the model. Must be implemented by derived classes.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples,)
            Target labels.
        """
        pass
    
    @abstractmethod
    def predict(self, X_test: np.array) -> np.array:
        """
        Summary
        -------
        Abstract method for predicting labels. Must be implemented by derived classes.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted labels for the test data.
        """
        pass
    
    def evaluate(self, X_test: np.array, y_test: np.array, name_model=None):
        """
        Summary
        -------
        Evaluates the model using test data and computes metrics. Saves the confusion matrix and classification report images.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        y_test : array-like of shape (n_samples,)
            True labels for the test data.
        name_model : str, optional
            Name of the model, used for naming the output directory (default is None).

        Returns
        -------
        None
        """
        predictions = self.predict(X_test)
        metrics = compute_metrics(predictions, y_test)
        output_dir = f"test_log/ML/{name_model}"
        os.makedirs(output_dir, exist_ok=True)
        save_confusion_matrix(metrics['confusion_matrix'], ['Class 0', 'Class 1', 'Class 2'], output_dir, -1, name_model)
        save_classification_report(metrics['classification_report'], output_dir, -1, name_model)


class KNN(Model_ML):
    """
    Summary
    -------
    The `KNN` class implements a K-Nearest Neighbors (KNN) model extending from a base machine learning abstract class `Model_ML`.
    It includes methods for training, prediction, and hyperparameter tuning using grid search.

    Attributes
    ----------
    model : KNeighborsClassifier
        An instance of `KNeighborsClassifier` from `sklearn.neighbors` initialized with specified hyperparameters.

    Methods
    -------
    train(X_train, y_train)
        Trains the KNN model using the provided training data.
    predict(X_test)
        Predicts the labels for the provided test data.
    grid_search(x_train, y_train)
        Performs a grid search to find the best hyperparameters for the KNN model.
    """
    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        """
        Summary
        -------
        Initializes the KNN model with specified hyperparameters.

        Parameters
        ----------
        n_neighbors : int, optional
            Number of neighbors to use (default is 5).
        weights : str, optional
            Weight function used in prediction (default is 'uniform').
        p : int, optional
            Power parameter for the Minkowski metric (default is 2).
        """
        super().__init__()                                                                  # Call the superclass constructor
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)    # Create a KNeighborsClassifier instance with specified hyperparameters.
    
    def train(self, X_train: np.array, y_train: np.array):
        """
        Summary
        -------
        Trains the KNN model using the provided training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples,)
            Target labels.
        """
        self.model.fit(X_train, y_train)                                                    # Fit the KNN model to the training data.
    
    def predict(self, X_test: np.array):
        """
        Summary
        -------
        Predicts the labels for the provided test data.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted labels for the test data.
        """
        return self.model.predict(X_test)                                                   # Predict labels for the test data.
    
    @staticmethod
    def grid_search(x_train: np.array, y_train: np.array) -> dict:
        """
        Summary
        -------
        Performs a grid search to find the best hyperparameters for the KNN model.

        Parameters
        ----------
        x_train : array-like of shape (n_samples, n_features) / np.array
            Training data.
        y_train : array-like of shape (n_samples) / np.array
            Target labels.

        Returns
        -------
        dict
            The best hyperparameters found by the grid search.
        """
        print ("[i] Computing Grid Search SVM .... ")               # Inform the user that grid search is starting.
        param_grid = {                                              # Define the parameter grid for grid search
            'n_neighbors': [2, 5, 100, 200, 500],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }

        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1, scoring='accuracy', return_train_score=True)                # Perform grid search with 5-fold cross-validation.
        grid_search.fit(x_train, y_train)                                                                                                           # Fit the grid search model to the training data.
        print("[i]KNN Best Parameters found:")
        for parameter, value in grid_search.best_params_.items():  # Inform the user of the best parameters found.
            print(f"└── {parameter}: {value}")
        return grid_search.best_params_


class SVM(Model_ML):   
    """
    Summary
    -------
    The `SVM` class implements a Support Vector Machine (SVM) model extending from a base machine learning abstract class `Model_ML`.
    It includes methods for training, prediction, and hyperparameter tuning using grid search.

    Attributes
    ----------
    model : SVC
        An instance of `SVC` from `sklearn.svm` initialized with specified hyperparameters.

    Methods
    -------
    train(X_train, y_train)
        Trains the SVM model using the provided training data.
    predict(X_test)
        Predicts the labels for the provided test data using sklearn predict method
    grid_search(x_train, y_train)
        Performs a grid search to find the best hyperparameters for the SVM model.

    """ 
    def __init__(self, C=1.0, kernel='linear', gamma='scale'):
        """
        Summary
        -------
        Initializes the SVM model with specified or default hyperparameters.

        Parameters
        ----------
        C : float, optional
            Regularization parameter (default is 1.0).
        kernel : str, optional
            Specifies the kernel type to be used in the algorithm (default is 'linear').
        gamma : str, optional
            Kernel coefficient for 'rbf', 'poly', and 'sigmoid' (default is 'scale').
        """
        super().__init__()
        self.model = SVC(max_iter=250)#(C=C, kernel=kernel, gamma=gamma)               # Define SVM model with the SVC istance
    
    def train(self, X_train: np.array, y_train: np.array):
        """
        Summary
        -------
        Trains the SVM model using the provided training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features) np.array
            Training data.
        y_train : array-like of shape (n_samples,) np.array
            Target labels.
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test: np.array) -> np.array:
        """
        Summary
        -------
        Predicts the labels for the provided test data.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted labels for the test data.
        """
        return self.model.predict(X_test)                       # Call predict method of sklearn.svm class
    
    @staticmethod
    def grid_search(x_train: np.array, y_train: np.array) -> dict:
        """
        Summary
        -------
        Performs a grid search to find the best hyperparameters for the SVM model.

        Parameters
        ----------
        x_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        dict
            The best hyperparameters found by the grid search.
        """
        print ("[i] Computing Grid Search SVM .... ")                                       # Inform the user that grid search is starting.
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': ['scale', 'auto'] + [0.1, 0.01, 0.001, 0.0001],
            'kernel': ['linear', 'poly'],# 'rbf', 'sigmoid'],
            'probability': [True, False],
            'tol': [0.0001, 0.001, 0.01, 0.1],
            'class_weight': ['balanced'],
            'break_ties': [True, False]
        }

        grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, return_train_score=True) # Perform grid search with 5-fold cross-validation
        grid_search.fit(x_train, y_train)                                                       # Fit the grid search model to the training data.
        print("[i]SVM Best Parameters found:")
        for parameter, value in grid_search.best_params_.items():                               # Print to the user the best parameter of grid search
            print(f"└── {parameter}: {value}")
        return grid_search.best_params_
