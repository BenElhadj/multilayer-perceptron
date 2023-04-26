import logging
import numpy as np

eps = 1e-17

def is_overfitting(losses, window_size=5):
    """
    Détermine si le modèle est en train de sur-apprendre en vérifiant si la perte augmente.
    Args:
        losses (list): Une liste de flottants représentant les pertes du modèle.
        window_size (int): La taille de la fenêtre pour calculer la perte moyenne.
    Returns:
        bool: True si la perte augmente, False sinon.
    """
    if len(losses) < window_size:
        return False
    return all(losses[i] <= losses[i + 1] for i in range(-window_size, -1))

class CrossEntropyLoss():
    """
    Classe pour calculer la perte de la fonction d'entropie croisée.
    La fonction d'entropie croisée est une mesure de l'écart entre la distribution de probabilité prédite et la distribution 
    de probabilité réelle. Elle est souvent utilisée pour les problèmes de classification.
    Cette classe fournit deux méthodes : loss et loss_derivative.
    """
    def loss(self, y_hat, y):
        """
        Calcule la perte de la fonction d'entropie croisée.
        Args:
            y_hat (np.ndarray): Tableau numpy de taille (m, n_classes) contenant les prédictions du modèle.
            y (np.ndarray): Tableau numpy de taille (m, n_classes) contenant les vraies étiquettes.
        Returns:
            float: La valeur de la perte de la fonction d'entropie croisée.
        """
        p = y
        q = y_hat
        logq = np.log(q + eps)
        return np.mean(-1.0 * np.sum(p * logq, axis=1))

    def loss_derivative(self, y_hat, y):
        """
        Calcule la dérivée de la perte de la fonction d'entropie croisée.
        Args:
            y_hat (np.ndarray): Tableau numpy de taille (m, n_classes) contenant les prédictions du modèle.
            y (np.ndarray): Tableau numpy de taille (m, n_classes) contenant les vraies étiquettes.
        Returns:
            np.ndarray: Tableau numpy de taille (m, n_classes) contenant la dérivée de la perte.
        """
        djda = -1 * y / (y_hat + eps)
        djda = djda / y.shape[0]
        return djda

class Optimizer():
    def __init__(self, learning_rate=0.1, Loss : CrossEntropyLoss = CrossEntropyLoss()):
        """
        Initialise un optimiseur avec un taux d'apprentissage et une fonction de perte.
        Args:
            learning_rate (float): Le taux d'apprentissage utilisé pour mettre à jour les poids des couches.
            Loss (CrossEntropyLoss): La fonction de perte à utiliser pour calculer l'erreur de la sortie du réseau de neurones.
        """
        self.Loss = Loss
        self.lr = learning_rate
        self.last_grad = None

    def update_weights(self, gradient, layer):
        """
        Met à jour les poids de la couche.
        Args:
            gradient (ndarray): Le gradient calculé pendant la rétropropagation.
            layer (Layer): La couche à mettre à jour.
        Returns:
            None
        """
        layer.w -= self.lr * gradient

    def fit(self, layers, y):
        """
        Entraîne le réseau de neurones en utilisant la rétropropagation.
        Args:
            layers (list): La liste des couches du réseau de neurones.
            y (ndarray): Le vecteur cible à utiliser pour entraîner le réseau de neurones.
        Returns:
            None
        """
        djda = self.Loss.loss_derivative(layers[-1].a, y)
        for l in reversed(layers):
            djda, weights_gradient = l.backpropagation(djda=djda)
            self.update_weights(weights_gradient, l)

class NAGOptimizer():
    def __init__(self, learning_rate=0.03, momentum=0.9, Loss=CrossEntropyLoss()):
        """
        Initialisation de l'optimiseur Nesterov Accelerated Gradient (NAG)
        Args:
        - learning_rate (float): taux d'apprentissage (default: 0.03)
        - momentum (float): coefficient d'inertie (default: 0.9)
        - Loss (objet): objet perte à minimiser (default: CrossEntropyLoss())
        """
        self.Loss = Loss
        self.lr = learning_rate
        self.local_gradient = 0
        self.momentum = momentum
        self.velocity = None
        self.optimizerlogger = logging.getLogger('Optimizer')
        self.optimizerlogger.setLevel(logging.WARNING)

    def update_weights(self, gradient, layer):
        """
        Met à jour les poids d'une couche avec le gradient correspondant
        Args:
        - gradient (ndarray): gradient de la couche
        - layer (objet): couche à mettre à jour
        """
        layer.w = layer.w - self.lr * gradient
        self.optimizerlogger.debug(f'l.w :\n{layer.w}')

    def apply_momentum_to_weights(self, layers):
        """
        Applique le coefficient d'inertie aux poids des couches
        """
        for l, v in zip(reversed(layers), self.velocity):
            l.w = l.w + self.momentum * v

    def fit(self, layers, y):
        """
        Entraîne le modèle sur un batch de données et de labels 
        Args:
        - layers (liste d'objets): couches du modèle
        - y (ndarray): labels correspondants au batch
        """
        self.local_gradient = self.Loss.loss_derivative(layers[-1].a, y)
        self.optimizerlogger.debug(f'loss dev: {self.local_gradient}')
        for l, i in zip(reversed(layers), range(len(layers))):
            self.local_gradient, weights_gradient = l.backpropagation(djda=self.local_gradient)
            self.optimizerlogger.debug(f'{self.local_gradient =}')
            self.update_weights(weights_gradient, l)
            self.velocity[i] = self.momentum * self.velocity[i] - weights_gradient * self.lr

    def pre_fit(self, x, y):
        """
        Prépare le modèle pour l'entraînement
        Args:
        - x (ndarray): batch de données
        - y (ndarray): batch de labels
        """
        if self.velocity is None:
            self.velocity = [0] * len(self.layers)
        self.apply_momentum_to_weights(self.layers)
