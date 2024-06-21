from abc import ABC, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd


class ML(ABC):
    """Classe base astratta per gli algoritmi di machine learning."""
    
    def __init__(self):
        self.model = None
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Metodo astratto per addestrare il modello."""
        pass
    
    @abstractmethod
    def predict(self, X_test):
        """Metodo astratto per fare previsioni."""
        pass
    
    def evaluate(self, X_test, y_test):
        """Valuta il modello utilizzando l'accuratezza."""
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)

class KNN(ML):
    """Implementazione dell'algoritmo K-Nearest Neighbors."""
    
    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        super().__init__()
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    
    def train(self, X_train, y_train):
        """Addestra il modello KNN."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Prevede i risultati con il modello KNN."""
        return self.model.predict(X_test)
    
    @staticmethod
    def grid_search(x_train, y_train):
        """Esegue una GridSearch per ottimizzare i parametri del modello KNN."""
        param_grid = {
            'n_neighbors': [2, 5, 100, 200, 500],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }

        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1, scoring='accuracy', return_train_score=True)
        grid_search.fit(x_train, y_train)
        return grid_search.best_params_


class SVM(ML):
    """Implementazione dell'algoritmo Support Vector Machine."""
    
    def __init__(self, C=1.0, kernel='linear', gamma='scale'):
        super().__init__()
        self.model = SVC(C=C, kernel=kernel, gamma=gamma)
    
    def train(self, X_train, y_train):
        """Addestra il modello SVM."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Prevede i risultati con il modello SVM."""
        return self.model.predict(X_test)
    
    @staticmethod
    def grid_search(x_train, y_train):
        """Esegue una GridSearch per ottimizzare i parametri del modello SVM."""
        param_grid = {
            #'C': [0.1, 1, 10, 100, 1000],
            #'gamma': ['scale', 'auto'] + [0.1, 0.01, 0.001, 0.0001],
            'kernel': ['linear', 'poly']# 'rbf', 'sigmoid'],
            #'probability': [True, False],
            #'tol': [0.0001, 0.001, 0.01, 0.1],
            #'class_weight': ['balanced'],
            #'break_ties': [True, False]
        }

        grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, return_train_score=True)
        grid_search.fit(x_train, y_train)
        return grid_search.best_params_


# Esempio di utilizzo:
if __name__ == "__main__":
    train_file = 'train_SVM.csv'
    test_file = 'test_SVM.csv'

    train_ds = pd.read_csv(train_file)
    test_ds = pd.read_csv(test_file)

    X_train, y_train = train_ds.iloc[:, :-1], train_ds.iloc[:, -1]
    X_test, y_test = test_ds.iloc[:, :-1], test_ds.iloc[:, -1]

    best_params_knn = KNN.grid_search(X_train, y_train)
    #best_params_svm = SVM.grid_search(X_train, y_train)

    print("Best parameters found for KNN:", best_params_knn)
    #print("Best parameters found for SVM:", best_params_svm)

    knn = KNN(**best_params_knn)
    knn.train(X_train, y_train)
    accuracy_knn = knn.evaluate(X_test, y_test)
    print("Accuracy of KNN:", accuracy_knn)

    #svm = SVM(**best_params_svm)
    svm = SVM()
    svm.train(X_train, y_train)
    accuracy_svm = svm.evaluate(X_test, y_test)
    print("Accuracy of SVM:", accuracy_svm)
