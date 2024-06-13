# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from cvxopt import matrix, solvers
# from sklearn.metrics import confusion_matrix, classification_report
# import itertools

# class SVM:
#     '''
#     This class implements a Support Vector Machine for classification from scratch using the cvxopt library.
#     '''
    
#     def __init__(self, C=1.0, kernel='linear', degree=3, gamma=None, coef0=0.0):
#         '''
#         Initializes the SVM model.
#         Args:
#             C: float, optional
#                 Penalty parameter C of the error term.
#             kernel: string, optional
#                 Specifies the kernel type to be used in the algorithm.
#             degree: int, optional
#                 Degree of the polynomial kernel function ('poly').
#             gamma: float, optional
#                 Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
#             coef0: float, optional
#                 Independent term in kernel function.
#         '''
#         self.C = C
#         self.kernel = kernel
#         self.degree = degree
#         self.gamma = gamma
#         self.coef0 = coef0
#         self.lagr_multipliers = None
#         self.support_vectors = None
#         self.support_vector_labels = None
#         self.intercept = None
        
#     def fit(self, X, y):
#         '''
#         Fits the SVM model according to the given training data.
#         Args:
#             X: numpy array
#                 Feature matrix.
#             y: numpy array
#                 Array of labels.
#         '''
#         n_samples, n_features = np.shape(X)
        
#         if not self.gamma:
#             self.gamma = 1 / n_features
            
#         if self.kernel == 'linear':
#             kernel = self.linear_kernel
#         elif self.kernel == 'rbf':
#             kernel = self.rbf_kernel
#         else:
#             raise ValueError("Invalid kernel type. Only 'linear' and 'rbf' are supported.")
        
#         # Gram matrix with jitter
#         K = np.zeros((n_samples, n_samples))
#         for i in range(n_samples):
#             for j in range(n_samples):
#                 K[i,j] = kernel(X[i], X[j])
#         K += np.eye(n_samples) * 1e-10  # Add jitter to the diagonal
                
#         # Define the quadratic optimization problem
#         P = matrix(np.outer(y, y) * K, tc='d')
#         q = matrix(-1 * np.ones(n_samples), tc='d')
#         A = matrix(y, (1, n_samples), tc='d')
#         b = matrix(0, tc='d')
        
#         if not self.C:
#             G = matrix(np.diag(np.ones(n_samples) * -1), tc='d')
#             h = matrix(np.zeros(n_samples), tc='d')
#         else:
#             G_max = np.diag(np.ones(n_samples) * -1)
#             G_min = np.identity(n_samples)
#             G = matrix(np.vstack((G_max, G_min)), tc='d')
#             h_max = matrix(np.zeros(n_samples), tc='d')
#             h_min = matrix(np.ones(n_samples) * self.C, tc='d')
#             h = matrix(np.vstack((h_max, h_min)), tc='d')
            
#         # Solve the quadratic optimization problem using cvxopt
#         minimization = solvers.qp(P, q, G, h, A, b)
#         lagr_mult = np.ravel(minimization['x'])

#         # Extract the support vectors
#         idx = lagr_mult > 1e-5
#         self.lagr_multipliers = lagr_mult[idx]
#         self.support_vectors = X[idx]
#         self.support_vector_labels = y[idx]
        
#         # Calculate the intercept
#         self.intercept = self.support_vector_labels[0]
        
#         for i in range(len(self.lagr_multipliers)):
#             self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[i] * kernel(self.support_vectors[i], self.support_vectors[0])
            
#     def predict(self, X):
#         '''
#         Predicts the classification of new points.
#         Args:
#             X: numpy array
#                 Feature matrix.
#         Returns:
#             numpy array
#                 Predicted class label per sample.
#         '''
        
#         if self.kernel == 'linear':
#             kernel = self.linear_kernel
#         elif self.kernel == 'rbf':
#             kernel = self.rbf_kernel
#         else:
#             raise ValueError("Invalid kernel type. Only 'linear' and 'rbf' are supported.")
        
#         y_pred = []
#         for sample in X:
#             prediction = 0
#             for i in range(len(self.lagr_multipliers)):
#                 prediction += self.lagr_multipliers[i] * self.support_vector_labels[i] * kernel(self.support_vectors[i], sample)
#             prediction += self.intercept
#             y_pred.append(np.sign(prediction))
#         return np.array(y_pred)
    
#     def linear_kernel(self, x1, x2):
#         '''
#         Linear kernel function.
#         Args:
#             x1: numpy array
#                 Sample 1.
#             x2: numpy array
#                 Sample 2.
#         Returns:
#             float
#                 Dot product of x1 and x2.
#         '''
#         return np.dot(x1, x2)
    
#     def rbf_kernel(self, x1, x2):
#         '''
#         Radial Basis Function (RBF) kernel function.
#         Args:
#             x1: numpy array
#                 Sample 1.
#             x2: numpy array
#                 Sample 2.
#         Returns:
#             float
#                 RBF similarity between x1 and x2.
#         '''
#         distance = np.linalg.norm(x1 - x2) ** 2
#         return np.exp(-self.gamma * distance)
    
#     def margin(self, X):
#         '''
#         Calculate margin for input data.
#         Args:
#             X: numpy array
#                 Feature matrix.
#         Returns:
#             numpy array
#                 Margin values.
#         '''
#         margins = []
#         for sample in X:
#             margin = 0
#             for i in range(len(self.lagr_multipliers)):
#                 margin += self.lagr_multipliers[i] * self.support_vector_labels[i] * self.rbf_kernel(self.support_vectors[i], sample)
#             margin += self.intercept
#             margins.append(margin)
#         return np.array(margins)

# def load_data(train_path, test_path, val_path):
#     '''
#     Load and preprocess the data from CSV files.
#     Args:
#         train_path: string
#             Path to the training data CSV file.
#         test_path: string
#             Path to the test data CSV file.
#         val_path: string
#             Path to the validation data CSV file.
#     Returns:
#         tuple
#             Training, validation, and test feature matrices and labels.
#     '''
#     def preprocess_data(df):
#         X = df.iloc[:, 2:32].values  # Select only the feature columns (2 to 31)
#         y = df.iloc[:, 32].values  # Select the label column (32)
#         return X, y
    
#     train_df = pd.read_csv(train_path)
#     test_df = pd.read_csv(test_path)
#     val_df = pd.read_csv(val_path)
    
#     X_train, y_train = preprocess_data(train_df)
#     X_test, y_test = preprocess_data(test_df)
#     X_val, y_val = preprocess_data(val_df)
    
#     return X_train, y_train, X_test, y_test, X_val, y_val

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     '''
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     '''
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# if __name__ == "__main__":
#     # Paths to the CSV files
#     train_path = 'train_SVM.csv'
#     test_path = 'test_SVM.csv'
#     val_path = 'val_SVM.csv'
#     X_train, y_train, X_test, y_test, X_val, y_val = load_data(train_path, test_path, val_path)

#     # Train the SVM model with linear kernel
#     svm_linear = SVM(kernel='linear')
#     svm_linear.fit(X_train, y_train)
#     y_val_pred_linear = svm_linear.predict(X_val)
#     y_test_pred_linear = svm_linear.predict(X_test)

#     # Train the SVM model with RBF kernel
#     svm_rbf = SVM(kernel='rbf')
#     svm_rbf.fit(X_train, y_train)
#     y_val_pred_rbf = svm_rbf.predict(X_val)
#     y_test_pred_rbf = svm_rbf.predict(X_test)

#     # Calculate and print accuracy
#     print("Validation accuracy (Linear Kernel):", np.mean(y_val_pred_linear == y_val))
#     print("Test accuracy (Linear Kernel):", np.mean(y_test_pred_linear == y_test))
#     print("Validation accuracy (RBF Kernel):", np.mean(y_val_pred_rbf == y_val))
#     print("Test accuracy (RBF Kernel):", np.mean(y_test_pred_rbf == y_test))

#     # Calculate and print classification report and confusion matrix
#     print("\nClassification Report (Linear Kernel):")
#     print(classification_report(y_test, y_test_pred_linear))
#     print("\nClassification Report (RBF Kernel):")
#     print(classification_report(y_test, y_test_pred_rbf))

#     # Confusion matrix for linear kernel
#     cm_linear = confusion_matrix(y_test, y_test_pred_linear)
#     print("\nConfusion Matrix (Linear Kernel):")
#     plot_confusion_matrix(cm_linear, classes=np.unique(y_test), title='Confusion Matrix (Linear Kernel)')

#     # Confusion matrix for RBF kernel
#     cm_rbf = confusion_matrix(y_test, y_test_pred_rbf)
#     print("\nConfusion Matrix (RBF Kernel):")
#     plot_confusion_matrix(cm_rbf, classes=np.unique(y_test), title='Confusion Matrix (RBF Kernel)')

#     plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from cvxopt import matrix, solvers
from sklearn.metrics import confusion_matrix, classification_report
import itertools

"""
Created on Sat Jun 1 12:43 CET 2024

@author: andreapietro.arena@unikorestudent.it

Some description
"""

class SVM:
    '''
    This class implements a Support Vector Machine for classification from scratch using the cvxopt library.
    '''
    
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma=None, coef0=0.0):
        '''
        Initializes the SVM model.
        Args:
            C: float, optional
                Penalty parameter C of the error term.
            kernel: string, optional
                Specifies the kernel type to be used in the algorithm.
            degree: int, optional
                Degree of the polynomial kernel function ('poly').
            gamma: float, optional
                Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
            coef0: float, optional
                Independent term in kernel function.
        '''
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None
        
    def fit(self, X, y):
        '''
        Fits the SVM model according to the given training data.
        Args:
            X: numpy array
                Feature matrix.
            y: numpy array
                Array of labels.
        '''
        n_samples, n_features = np.shape(X)
        
        if not self.gamma:
            self.gamma = 1 / n_features
            
        if self.kernel == 'linear':
            kernel = self.linear_kernel
        elif self.kernel == 'rbf':
            kernel = self.rbf_kernel
        else:
            raise ValueError("Invalid kernel type. Only 'linear' and 'rbf' are supported.")
        
        # Gram matrix with jitter
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = kernel(X[i], X[j])
        K += np.eye(n_samples) * 1e-10  # Add jitter to the diagonal
                
        # Define the quadratic optimization problem
        P = matrix(np.outer(y, y) * K, tc='d')
        q = matrix(-1 * np.ones(n_samples), tc='d')
        A = matrix(y, (1, n_samples), tc='d')
        b = matrix(0, tc='d')
        
        if not self.C:
            G = matrix(np.diag(np.ones(n_samples) * -1), tc='d')
            h = matrix(np.zeros(n_samples), tc='d')
        else:
            G_max = np.diag(np.ones(n_samples) * -1)
            G_min = np.identity(n_samples)
            G = matrix(np.vstack((G_max, G_min)), tc='d')
            h_max = matrix(np.zeros(n_samples), tc='d')
            h_min = matrix(np.ones(n_samples) * self.C, tc='d')
            h = matrix(np.vstack((h_max, h_min)), tc='d')
        
        # Solve the quadratic optimization problem using cvxopt
        try:
            minimization = solvers.qp(P, q, G, h, A, b)
        except ValueError as e:
            print("Errore durante la risoluzione del problema di ottimizzazione:", e)
            return
        
        lagr_mult = np.ravel(minimization['x'])

        # Extract the support vectors
        idx = lagr_mult > 1e-5
        self.lagr_multipliers = lagr_mult[idx]
        self.support_vectors = X[idx]
        self.support_vector_labels = y[idx]
        
        # Calculate the intercept
        self.intercept = self.support_vector_labels[0]
        
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[i] * kernel(self.support_vectors[i], self.support_vectors[0])
            
    def predict(self, X):
        '''
        Predicts the classification of new points.
        Args:
            X: numpy array
                Feature matrix.
        Returns:
            numpy array
                Predicted class label per sample.
        '''
        
        if self.kernel == 'linear':
            kernel = self.linear_kernel
        elif self.kernel == 'rbf':
            kernel = self.rbf_kernel
        else:
            raise ValueError("Invalid kernel type. Only 'linear' and 'rbf' are supported.")
        
        y_pred = []
        for sample in X:
            prediction = 0
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[i] * kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)
    
    def linear_kernel(self, x1, x2):
        '''
        Linear kernel function.
        Args:
            x1: numpy array
                Sample 1.
            x2: numpy array
                Sample 2.
        Returns:
            float
                Dot product of x1 and x2.
        '''
        return np.dot(x1, x2)
    
    def rbf_kernel(self, x1, x2):
        '''
        Radial Basis Function (RBF) kernel function.
        Args:
            x1: numpy array
                Sample 1.
            x2: numpy array
                Sample 2.
        Returns:
            float
                RBF similarity between x1 and x2.
        '''
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-self.gamma * distance)
    
    def margin(self, X):
        '''
        Calculate margin for input data.
        Args:
            X: numpy array
                Feature matrix.
        Returns:
            numpy array
                Margin values.
        '''
        margins = []
        for sample in X:
            margin = 0
            for i in range(len(self.lagr_multipliers)):
                margin += self.lagr_multipliers[i] * self.support_vector_labels[i] * self.rbf_kernel(self.support_vectors[i], sample)
            margin += self.intercept
            margins.append(margin)
        return np.array(margins)

def load_data(train_path, test_path, val_path):
    '''
    Load and preprocess the data from CSV files.
    Args:
        train_path: string
            Path to the training data CSV file.
        test_path: string
            Path to the test data CSV file.
        val_path: string
            Path to the validation data CSV file.
    Returns:
        tuple
            Training, validation, and test feature matrices and labels.
    '''
    def preprocess_data(df, scaler=None):
        X = df.iloc[:, 2:32].values  # Select only the feature columns (2 to 31)
        y = df.iloc[:, 32].values  # Select the label column (32)
        if scaler:
            X = scaler.transform(X)
        return X, y
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)
    
    # Normalize the training data
    scaler = MinMaxScaler()
    X_train, y_train = preprocess_data(train_df)
    X_train = scaler.fit_transform(X_train)
    
    # Normalize the test and validation data using the same scaler
    X_test, y_test = preprocess_data(test_df, scaler=scaler)
    X_val, y_val = preprocess_data(val_df, scaler=scaler)
    
    return X_train, y_train, X_test, y_test, X_val, y_val

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":
    # Paths to the CSV files
    train_path = 'train_SVM.csv'
    test_path = 'test_SVM.csv'
    val_path = 'val_SVM.csv'
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(train_path, test_path, val_path)
    
    # Initialize the SVM model
    svm = SVM(C=1.0, kernel='rbf', gamma=0.1)
    
    # Fit the model to the training data
    svm.fit(X_train, y_train)
    
    # Predict on the validation data
    y_val_pred = svm.predict(X_val)
    
    # Evaluate the model
    print("Confusion Matrix:")
    cm = confusion_matrix(y_val, y_val_pred)
    plot_confusion_matrix(cm, classes=[1, 2, 3], title='Confusion Matrix')
    
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Class 1', 'Class 2', 'Class 3']))
