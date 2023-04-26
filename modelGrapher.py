import os
import pickle
import shutil
from matplotlib import animation, gridspec, pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt
import seaborn as sns
from loss import CrossEntropyLoss
from dataset import Dataset
from dataset import BatchIterator
from loss import NAGOptimizer
import layer

# Model
eps = 1e-17

class Model():
    """
    Modèle de réseau de neurones artificiels.
    """
    optimizer = NAGOptimizer(learning_rate = 0.05, momentum = 0.7)
    def __init__(self, sizes, activations, optimizer):
        """
        Initialise le modèle.
        Args:
            sizes (List[int]): Liste des tailles de chaque couche de neurones.
            activations (List[str]): Liste des noms des fonctions d'activation pour chaque couche.
            optimizer (Optimizer): Optimiseur utilisé pour l'entraînement du modèle.
        """
        self.layers = layer.make_layer_list_from_sizes_and_activations(sizes, activations)
        self.Optimizer = optimizer
        self.Optimizer.layers = self.layers
        self.grapher = Grapher()
        self.sizes = sizes
        self.activations = activations

    def feedforward(self, x):
        """
        Effectue une propagation avant dans le réseau de neurones.
        Args:
            x (ndarray): Entrée du réseau de neurones.
        Returns:
            ndarray: Sortie du réseau de neurones.
        """
        for l in self.layers:
            x = l.forward(x)
        return x

    def fit(self, x, y):
        """
        Effectue une itération de l'entraînement du modèle.
        Args:
            x (ndarray): Entrée du réseau de neurones.
            y (ndarray): Sortie attendue du réseau de neurones.
        """
        self.Optimizer.pre_fit(x, y)
        self.feedforward(x)
        self.Optimizer.fit(self.layers, y)

    def epoch(self, batch_iterator):
        """
        Effectue une époque d'entraînement du modèle.
        Args:
            batch_iterator (BatchIterator): Itérateur de mini-batchs.
        """
        for x, y in batch_iterator:
            self.fit(x, y)

    def train(self, dataset, batch_size=0):
        """
        Entraîne le modèle sur un ensemble de données.
        Args:
            dataset (Dataset): Ensemble de données d'entraînement.
            batch_size (int): Taille des mini-batchs. Si batch_size = 0, utilise l'ensemble de données complet.
        """
        batch_iterator = BatchIterator(dataset.x, dataset.y, batch_size)
        for x, y in batch_iterator:
            self.fit(x, y)

    def save(self, path):
        """
        Sauvegarde les paramètres du modèle.
        Args:
            path (str): Chemin du dossier de sauvegarde.
        """
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        np.savetxt(os.path.join(path, 'activations.csv'), self.activations, delimiter=',', fmt='%s')
        np.savetxt(os.path.join(path, 'sizes.csv'), self.sizes, delimiter=',', fmt='%d')
        with open(os.path.join(path, 'optimizer.pkl'), 'wb') as f:
            pickle.dump(self.Optimizer, f)
        for i, layer in enumerate(self.layers):
            np.savetxt(os.path.join(path, f'weights_{i}.csv'), layer.w, delimiter=',')
        self.grapher.metrics.to_csv(os.path.join(path, 'metrics.csv'))

    def __str__(self):
        """
        Renvoie une chaîne de caractères décrivant le modèle.
        """
        return '   '.join([str(l) for l in self.layers])

# Grapher

def evaluate_binary_classifier(model, x, y):
    """
    Évalue les prédictions du modèle pour une classification binaire.
    Args:
        model (NeuralNetwork): Le modèle de réseau de neurones à évaluer.
        x (ndarray): Les données d'entrée.
        y (ndarray): Les étiquettes de sortie correspondantes.
    Returns:
        tuple: Un tuple contenant les nombres de vrais positifs, faux positifs, vrais négatifs et faux négatifs.
    """
    yhat = model.feedforward(x)
    e = (2 * y )+ (yhat == yhat.max(axis=1, keepdims = True)).astype(int)
    return (e[:, 1] == 3).astype(int).sum(), (e[:, 1] == 1).astype(int).sum(),\
        (e[:, 1] == 0).astype(int).sum(), (e[:, 1] == 2).astype(int).sum()

def binary_classification_metrics(tp, fp, tn, fn):
    """
    Calcul des métriques de classification pour un problème de classification binaire.
    Args:
    tp : int
        Le nombre de vrais positifs.
    fp : int
        Le nombre de faux positifs.
    tn : int
        Le nombre de vrais négatifs.
    fn : int
        Le nombre de faux négatifs.
    Returns:
    tuple
        Un tuple contenant la sensibilité, la spécificité, la précision et la F1-score.
    """
    sensitivity = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    return sensitivity, tn / (tn + fp + eps), precision, 2.0 * (sensitivity * precision) / (sensitivity + precision + eps)

def calculate_and_display_metrics(model, x, y):
    """
    Calcul et affichage des métriques de classification pour un modèle donné.
    Args:
    model : object
        Le modèle de réseau de neurones.
    x : array_like
        Les entrées du modèle.
    y : array_like
        Les sorties attendues.
    Returns:
    tuple
        Un tuple contenant la sensibilité, la spécificité, la précision et la F1-score.
    """
    yhat = model.feedforward(x)
    yhatmax = np.argmax(yhat, axis=1)
    ymax = np.argmax(y, axis=1)
    e = 2 * ymax + yhatmax
    tn = np.sum(e == 0)
    fp = np.sum(e == 1)
    fn = np.sum(e == 2)
    tp = np.sum(e == 3)
    sensitivity, specificity, precision, f1 = binary_classification_metrics(tp, fp, tn, fn)
    return (sensitivity, specificity, precision, f1)

class Grapher():
    """
    Classe pour tracer les métriques du modèle.
    Attributes:
        loss (CrossEntropyLoss): la fonction de perte utilisée pour l'entraînement du modèle.
        metrics (pandas.DataFrame): le tableau de bord des métriques qui sera rempli à chaque époque.
        epoch (int): le numéro de l'époque actuelle.
    Methods:
        __init__ : initialise les attributs de la classe.
        add_data_point : ajoute les métriques actuelles à la tableau de bord des métriques.
        plot_metrics : trace les courbes des métriques (perte et F1-score) sur un graphique.
        calculate_metrics : calcule la perte et F1-score pour l'ensemble d'entraînement et de test et les ajoute au tableau de bord.
        print_metrics : affiche les métriques actuelles (perte et perte de validation).

    """
    loss = CrossEntropyLoss()
    def __init__(self):
        """
        Initialise les attributs de la classe.
        """
        self.metrics = pd.DataFrame(columns=['Epoch', 'Loss', 'Validation Loss', 'F1'])
        sns.set_theme(style='darkgrid')
        self.epoch = 0

    def add_data_point(self, loss, val_loss, f1):
        """
        Ajoute les métriques actuelles à la tableau de bord des métriques.
        Args:
            loss (float): la perte actuelle sur l'ensemble d'entraînement.
            val_loss (float): la perte actuelle sur l'ensemble de validation.
            f1 (float or None): le score F1 actuel ou None si le modèle n'est pas un classifieur binaire.
        Returns:
            None
        """
        self.metrics.loc[self.epoch] = [self.epoch, loss, val_loss, f1]
        self.epoch += 1

    def plot_metrics(self, y_pred, y_true):
        """
        Trace les courbes des métriques (perte et F1-score) sur un graphique.
        Tracees de l'amelioration de la prediction au cours des époques.
        Returns:
            None
        """
        G = gridspec.GridSpec(2, 2)
        fig = plt.figure()
        ax1 = plt.subplot(G[0, :])

        # Plot du premier graphique (les métriques) dans le premier sous-graphique (ax1)
        
        def update(frame):
            ax1.clear()
            sns.lineplot(x='Epoch', y='value', data=pd.melt(self.metrics[self.metrics['Epoch'] <= frame], id_vars=['Epoch']),
                        hue='variable', legend='full', ax=ax1)
            ax1.set_title('Metrics and Predicted vs Actual')

        ani1 = animation.FuncAnimation(fig, update, frames=range(int(self.metrics['Epoch'].max())), repeat=False)

        # Plot du second graphique (les prédictions vs les valeurs réelles) dans le second sous-graphique (ax2)
        ax2 = plt.subplot(G[1,:-1])
        actual = np.argmax(y_true, axis=1)
        colors = ['chartreuse', 'red', 'green', 'maroon']
        def update(frame):
            ax2.clear()
            pred = np.argmax(y_pred[frame], axis=1)
            pred_counts = np.bincount(pred)
            actual_counts = np.bincount(actual)
            labels = []
            for i, count in enumerate(pred_counts):
                labels.append(f"Pred {'Malignant' if i else 'Benign'} = {count}")
            for i, count in enumerate(actual_counts):
                labels.append(f"Actual {'Malignant' if i else 'Benign'} = {count}")

            sc = ax2.scatter(y_pred[frame, :, 0], y_pred[frame, :, 1], c=[colors[0] if pred[i] == 0 else colors[1] for i in range(len(pred))], marker='.')
            sc = ax2.scatter(y_true[:, 0], y_true[:, 1], c=[colors[2] if actual[i] == 0 else colors[3] for i in range(len(actual))], marker='o')
            handles = [plt.scatter([],[], color=colors[i], label=labels[i]) for i in range(len(labels))]

            benign_max = np.max(y_pred[frame, pred == 0, 0]) if y_pred[frame, pred == 0].shape[0] > 0 else 0
            malignant_min = np.min(y_pred[frame, pred == 1, 0]) if y_pred[frame, pred == 1].shape[0] > 0 else 0

            if benign_max > 0 and malignant_min > 0:
                plt.plot([benign_max, malignant_min], [1, 0], 'black', linestyle='-', linewidth=0.5)
            else:
                plt.axvline(x=0.5, color='black', linestyle='-', linewidth=0.5)

            plt.legend(handles=handles, loc='upper right')

            plt.xlabel('1 = Benign')
            plt.ylabel('1 = Malignant')
            plt.title(f'Epoch {frame + 2}')

        ani2 = animation.FuncAnimation(fig, update, frames=range(y_pred.shape[0]), repeat=False)
        plt.show()

    def plot_predict(model,data, general_loss, sensitivity, specificity, precision, f1):
        y_pred = model.feedforward(data.x)
        y_true = np.argmax(data.y, axis=1)
        y_pred_axis1 = np.argmax(y_pred, axis=1)
        nb_pred_erreur = np.sum(y_true != y_pred_axis1)

        colors = ['chartreuse', 'red', 'green', 'maroon']
        labels = [f"y_true_benign={np.bincount(y_true)[0]}", f"y_true_malignant={np.bincount(y_true)[1]}", f"y_pred_benign={np.bincount(y_pred_axis1)[0]}", f"y_pred_malignant={np.bincount(y_pred_axis1)[1]}"]

        plt.scatter(y_pred[:, 0], y_pred[:, 1], c=[colors[0] if y_pred_axis1[i] == 0 else colors[1] for i in range(len(y_pred_axis1))], marker='.')
        
        plt.scatter(data.y[:, 0], data.y[:, 1], c=[colors[2] if y_true[i] == 0 else colors[3] for i in range(len(y_true))], marker='o')
        plt.title(f'Model Performance\nGeneral Loss: {general_loss:.4f}')
        plt.xlabel(f"Total erreur = {nb_pred_erreur}, {sensitivity = :.3f}, {specificity = :.3f}, {precision = :.3f}, {f1 = :.3f}\n", fontsize=10)

        handles = [plt.scatter([],[], color=colors[i], label=labels[i]) for i in range(len(colors))]
        plt.legend(handles=handles, loc='upper right')

        plt.show()
        return nb_pred_erreur

    def calculate_metrics(self, m: Model, train_d: Dataset, test_d: Dataset, d: Dataset) -> None:
        """
        Calcule la perte et F1-score pour l'ensemble d'entraînement et de test et les ajoute au tableau de bord.
        Args:
            m (Model): le modèle entraîné.
            train_d (Dataset): l'ensemble d'entraînement.
            test_d (Dataset): l'ensemble de test.
            d (Dataset): l'ensemble de données complet (pour calculer le score F1).
        Returns:
            None
        """
        tp, fp, tn, fn = evaluate_binary_classifier(m, d.x, d.y)
        _, _, _, f1 = binary_classification_metrics(tp, fp, tn, fn)
        loss = m.Optimizer.Loss.loss(m.feedforward(train_d.x), train_d.y)
        val_loss = m.Optimizer.Loss.loss(m.feedforward(test_d.x), test_d.y)
        self.add_data_point(loss, val_loss, f1)
        self.loss = loss
        self.val_loss = val_loss

    def print_metrics(self, iter):
        print(f'epoch: {self.epoch + 1:4}/{iter}  -  loss: {self.loss:.4f}  -  val_loss: {self.val_loss:.4f}', end='\r')
