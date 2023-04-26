import os
import pickle
import pandas as pd
import numpy as np
from math import ceil
from sklearn.preprocessing import StandardScaler

# Dataset

def create_dataset(path = 'data.csv', usecols = list(range(32)), y_col = 1, y_categorical = True):
    """
    Créé un ensemble de données à partir d'un fichier CSV.
    Args:
    - path (str): le chemin d'accès au fichier CSV.
    - usecols (list): la liste des indices des colonnes à inclure dans les données d'entrée.
    - y_col (int): l'indice de la colonne contenant les données de sortie.
    - y_categorical (bool): True si les données de sortie sont catégorielles, False sinon.
    Returns:
    - dataset (Dataset): l'ensemble de données créé.
    """
    df = pd.read_csv(path, usecols=usecols, header=None)
    df.fillna(df.median(numeric_only=True), inplace=True)
    if (y_categorical):
        y_df = pd.get_dummies(df[y_col] ,drop_first = False)
    else:
        y_df = df[y_col]
    df.drop([y_col], axis = 1, inplace = True)
    y = y_df.to_numpy()
    x = df.to_numpy()
    return Dataset(x, y, standardize = True)

class BatchIterator():
    def __init__(self, x, y, batch_size):
        """
        Initialise l'itérateur de batch avec les données x et y et la taille de batch spécifiée.
        Args:
        x -- tableau de forme (n_samples, n_features) contenant les données d'entraînement.
        y -- tableau de forme (n_samples,) contenant les étiquettes d'entraînement.
        batch_size -- taille du batch pour chaque itération.

        Attributs:
        x -- tableau de forme (n_samples, n_features) contenant les données d'entraînement.
        y -- tableau de forme (n_samples,) contenant les étiquettes d'entraînement.
        batch_size -- taille du batch pour chaque itération.
        size -- nombre total de données d'entraînement.
        batch_num_max -- nombre maximum de batches qui peuvent être générés.
        batch_num_current -- numéro de batch actuel.
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.size = self.x.shape[0]
        if ((self.batch_size <= 0) or (self.batch_size > self.size)):
            self.batch_size = self.size
        self.batch_num_max = int(ceil(self.size / self.batch_size))
        self.batch_num_current = -1
        self.shuffle()

    def shuffle(self):
        """
        Mélange les données d'entraînement et les étiquettes d'entraînement.
        """
        idx = np.random.permutation(self.size)
        self.x = self.x[idx]
        self.y = self.y[idx]

    def gen_ith_batch(self, i):
        """
        Génère le i-ème batch de données et d'étiquettes.
        Args:
        i -- numéro du batch.
        Retour:
        Un tuple contenant le i-ème batch de données et d'étiquettes.
        """
        start_idx = i * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.size)
        return self.x[start_idx:end_idx], self.y[start_idx:end_idx]

    def __iter__(self):
        """
        Retourne l'itérateur de batch.
        """
        self.current_k = -1
        return self

    def __next__(self):
        """
        Génère le prochain batch de données et d'étiquettes.
        Retour:
        Un tuple contenant le prochain batch de données et d'étiquettes.
        Lève:
        StopIteration -- si le dernier batch a été généré.
        """
        self.batch_num_current += 1

        if (self.batch_num_current >= self.batch_num_max):
            raise StopIteration
        else:
            return self.gen_ith_batch(self.batch_num_current)

class Dataset():
    def __init__(self, x, y, standardize=True):
        """
        Classe pour stocker et manipuler les données d'un dataset.
        Args:
            x (numpy.ndarray): Features (m x p).
            y (numpy.ndarray): Labels (m x y_p).
            standardize (bool, optional): Indique si les données doivent être standardisées. Defaults to True.
        """
        self.x = x
        self.y = y
        self.y_p = y.shape[1]
        self.p = x.shape[1]
        self.m = x.shape[0]
        if standardize:
            self.standardize()

    def standardize(self):
        """
        Standardise les features x.
        """
        self.x_scaler = StandardScaler()
        self.x = self.x_scaler.fit_transform(self.x)

    def batchiterator(self, batchsize):
        """
        Retourne un itérateur de mini-batches.
        Args:
            batchsize (int): Taille de chaque mini-batch.
        Returns:
            BatchIterator: Itérateur de mini-batches.
        """
        return BatchIterator(self.x, self.y, batchsize)

    def save_norm(self, model_name):
        """
        Enregistre la transformation de standardisation des features.
        Args:
            model_name (str): Nom du modèle.
        """
        path = os.path.join(model_name, 'norm.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.x_scaler, f)

class KFoldIterator():
    def __init__(self, x, y, k):
        """
        Initialisation d'un itérateur de validation croisée k-fold.
        Args:
            x (numpy.ndarray): Tableau de données d'entrée.
            y (numpy.ndarray): Tableau de données de sortie.
            k (int): Nombre de folds.
        Raises:
            ValueError: Si le nombre de folds est supérieur ou égal à la taille des données.
        """
        self.k = k
        self.x = x
        self.y = y
        self.size = x.shape[0]
        if self.k > self.size:
            raise ValueError(f'The number of folds must be smaller than {self.size - 1}.')
        self.idx = np.random.permutation(self.size)
        self.current_k = -1

    def gen_ith_fold(self, i):
        """
        Génère le i-ème fold de validation croisée.
        Args:
            i (int): Numéro du fold.
        Returns:
            tuple: Un tuple contenant deux objets Dataset, l'un pour les données d'entraînement et l'autre pour les données de test.
        """
        start = i * self.size // self.k
        end = (i + 1) * self.size // self.k
        test_mask = np.zeros(self.size, dtype=bool)
        test_mask[self.idx[start:end]] = True
        train_mask = ~test_mask
        x_train = self.x[train_mask]
        y_train = self.y[train_mask]
        x_test = self.x[test_mask]
        y_test = self.y[test_mask]
        return Dataset(x_train, y_train, standardize=False), Dataset(x_test, y_test, standardize=False)

    def __next__(self):
        """
        Renvoie le fold de validation croisée suivant.
        Returns:
            tuple: Un tuple contenant deux objets Dataset, l'un pour les données d'entraînement et l'autre pour les données de test.
        """
        self.current_k += 1
        if self.current_k >= self.k:
            raise StopIteration
        return self.gen_ith_fold(self.current_k)
