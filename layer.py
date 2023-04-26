import logging
import numpy as np
from math import sqrt
from scipy.special import expit

# Layer

def init_weights(in_size, out_size):
    """
    Initialise les poids pour une couche de réseau de neurones.
    Args:
    in_size (int): La taille de la couche d'entrée.
    out_size (int): La taille de la couche de sortie.
    Returns:
    np.ndarray: Un tableau 2D des poids initialisés aléatoirement.
    """
    invsqrtn = 1.0 / sqrt(in_size)
    return np.random.uniform(-invsqrtn, invsqrtn, (in_size + 1, out_size))

def make_layer_list_from_sizes_and_activations(sizes, activations):
    """
    Crée une liste de couches à partir des tailles des couches et des fonctions d'activation fournies.
    Args:
        sizes (list): Une liste d'entiers représentant les tailles des couches.
        activations (list): Une liste de chaînes de caractères représentant les fonctions d'activation
            pour chaque couche. Les fonctions d'activation possibles sont : 'sigmoid', 'tanh', 'relu', 'softmax'.
    Raises:
        ValueError: Si la longueur de la liste sizes est différente de la longueur de la liste activations + 1.
    Returns:
        list: Une liste d'instances Layer initialisées avec les tailles et les fonctions d'activation correspondantes.
    """
    if len(sizes) != len(activations) + 1:
        raise ValueError("Please enter an activation function for every layer")
    return [
        Layer(sizes[i - 1], size, activations[i - 1])
        for i, size in enumerate(sizes[1:], start=1)
    ]

def activation_sigmoid(z):
    """
    Calcule la fonction d'activation sigmoïde.
    Args:
    z (np.ndarray): Le produit matriciel des entrées et des poids.
    Returns:
    np.ndarray: Le résultat de l'application de la fonction sigmoïde à z.
    """
    return (expit(z))

def derivative_sigmoid(z, a):
    """
    Calcule la dérivée de la fonction d'activation sigmoïde.
    Args:
    z (np.ndarray): Le produit matriciel des entrées et des poids.
    a (np.ndarray): La sortie de la fonction d'activation sigmoïde pour z.
    Returns:
    np.ndarray: La matrice jacobienne de la fonction sigmoïde appliquée à z.
    """
    da = (a) * (1 - a)
    b, n = da.shape
    da = np.einsum('ij,jk->ijk' , da, np.eye(n, n))
    return da

def activation_softmax(z):
    """
    Calcule la fonction d'activation softmax.
    Args:
    z (np.ndarray): Le produit matriciel des entrées et des poids.
    Returns:
    np.ndarray: Le résultat de l'application de la fonction softmax à z.
    """
    z_max = np.max(z, axis=1, keepdims=True)
    z_exp = np.exp(z - z_max)
    z_sum = np.sum(z_exp, axis=1, keepdims=True)
    return z_exp / z_sum

def derivative_softmax(z, a):
    """
    Calcul de la dérivée de la fonction softmax.
    Args:
    z : array_like
        Les entrées de la couche.
    a : array_like
        Les sorties de la couche.
    Returns:
    array_like
        La matrice des dérivées de la fonction softmax.
    """
    m, n = a.shape
    t1 = np.einsum('ij,ik->ijk', a, a)
    diag = np.einsum('ik,jk->ijk', a, np.eye(n, n))
    return diag - t1

def derivative_identity(z, a):
    """
    Calcule la dérivée de la fonction d'activation identité.
    Args:
    z (np.ndarray): Le produit matriciel des entrées et des poids.
    a (np.ndarray): La sortie de la fonction d'activation identité pour z.
    Returns:
    np.ndarray: La matrice jacobienne de la fonction identité appliquée à z.
    """
    return np.eye(a.shape[1])

def derivative_RectifiedLinearUnit(z, a):
    """
    Calcule la dérivée de la fonction d'activation rectified linear unit (ReLU).
    Args:
    z (np.ndarray): Le produit matriciel des entrées et des poids.
    a (np.ndarray): La sortie de la fonction d'activation ReLU pour z.
    Returns:
    Tuple[np.ndarray, np.ndarray]: Le résultat de l'application de la fonction ReLU à z et sa matrice jacobienne.
    """
    _, n = z.shape
    dz = np.where(z > 0, 1, 0)
    return 0.5 * (np.abs(z) + z), np.outer(dz, np.eye(n))

def get_activation_functions(activation):
    """
    Obtention de la fonction d'activation et de sa dérivée pour une couche donnée.
    Args:
    activation : str
        Le nom de la fonction d'activation.
    Returns:
    tuple
        Un tuple contenant la fonction d'activation et sa dérivée.
    """
    ACTIVATIONS = {
        'sigmoid': (activation_sigmoid, derivative_sigmoid),
        'softmax': (activation_softmax, derivative_softmax),
        'identity': derivative_identity,
        'rectifiedLinearUnit': derivative_RectifiedLinearUnit
    }
    if activation not in ACTIVATIONS:
        print('You have entered an incorrect activation function name, defaulting to sigmoid')
        return ACTIVATIONS['sigmoid']
    return ACTIVATIONS[activation]

class Layer():
    """
    Classe représentant une couche d'un réseau de neurones.
    Cette classe est utilisée pour créer des couches qui seront ensuite utilisées
    pour construire un réseau de neurones.
    Attributes:
        activation (str): La fonction d'activation utilisée pour calculer les sorties de la couche.
        activation_derivative (function): La dérivée de la fonction d'activation utilisée pour le rétropropagation.
        w (ndarray): Les poids de la couche.
        local_gradient (ndarray): Le gradient local de la couche.
    Methods:
        __init__(self, in_size, out_size, activation='sigmoid'): Initialise une nouvelle couche avec des poids aléatoires.
        forward(self, x): Calcule les sorties de la couche pour une entrée donnée.
        backpropagation(self, djda): Calcule le gradient pour la couche précédente et les poids de la couche actuelle.
        __str__(self): Retourne une chaîne de caractères décrivant la taille des poids de la couche.
    """
    def __init__(self, in_size, out_size, activation = 'sigmoid'):
        """
        Initialise une nouvelle couche avec des poids aléatoires.
        Args:
            in_size (int): La taille de l'entrée de la couche.
            out_size (int): La taille de la sortie de la couche.
            activation (str, optional): Le nom de la fonction d'activation à utiliser. Par défaut, 'sigmoid' est utilisé.
        Returns:
            None
        """
        self.activation, self.activation_derivative = get_activation_functions(activation)
        self.w = init_weights(in_size, out_size)
        self.local_gradient = None

    def forward(self, x):
        """
        Calcule les sorties de la couche pour une entrée donnée.
        Args:
            x (ndarray): Les entrées de la couche.
        Returns:
            ndarray: Les sorties de la couche.
        """
        forwardlogger = logging.getLogger('FeedForward')
        forwardlogger.debug(f'x:\n{x.shape}\n w:\n{self.w.shape}\n')
        self.x = np.concatenate((np.ones([x.shape[0], 1]), x), axis = 1)
        self.z = np.matmul(self.x, self.w)
        self.a = self.activation(self.z)
        return (self.a)

    def backpropagation(self, djda):
        """
        Calcule le gradient pour la couche précédente et les poids de la couche actuelle.
        Args:
            djda (ndarray): Le gradient de l'erreur par rapport aux sorties de la couche actuelle.
        Returns:
            tuple: Un tuple contenant le gradient pour la couche précédente et les poids de la couche actuelle.
        """
        backproplogger = logging.getLogger('BackProp')
        backproplogger.debug(f'Backpropagation : {self.activation.__name__}')
        dadz = self.activation_derivative(self.z, self.a)
        backproplogger.debug(f'djda:\n {djda}\n')
        djdz = np.einsum( 'ik,ikj->ij', djda, dadz)
        backproplogger.debug(f'djdz:\n {djdz}\n')
        djdw = np.matmul(self.x.T, djdz)
        backproplogger.debug(f'djdw:\n{djdw}\n')
        next_djda = np.matmul(djdz, self.w[1:, :].T)
        backproplogger.debug(f'{next_djda.shape = } \nnext_djda:  \n{next_djda * 10}\n\n')
        return next_djda, djdw

    def __str__(self):
        """
        Renvoie une représentation en chaîne de caractères de la couche.
        Retourne:
            str : Une chaîne de caractères représentant la taille des poids de la couche.
        """
        return (f'w.shape: {self.w.shape}')