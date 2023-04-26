from matplotlib import pyplot as plt
from dataset import create_dataset
import modelGrapher
import pandas as pd
import numpy as np
import argparse
import pickle
import sys
import os

def load_model(model_name="model"):
    path = model_name
    activations = np.genfromtxt(os.path.join(path, 'activations.csv'), delimiter=",", dtype=str)
    sizes = np.genfromtxt(os.path.join(path, "sizes.csv"), delimiter=",", dtype=int)
    with open(os.path.join(path, "optimizer.pkl"), "rb") as f:
        Optimizer = pickle.load(f)
    model = modelGrapher.Model(sizes, activations, optimizer=Optimizer)
    for i in range(len(sizes) - 1):
        model.layers[i].w = np.genfromtxt(os.path.join(path, f"weights_{i}.csv"), delimiter=",")
    model.grapher.metrics = pd.read_csv(os.path.join(path, "metrics.csv"))

    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='A typical logistic regression', allow_abbrev=False)
    parser.add_argument('arg1', nargs='?', help="path to input data or model name")
    parser.add_argument('arg2', nargs='?', help="path to input data or model name")
    args = parser.parse_args()

    dataset_path = None
    model_name = None

    for arg in [args.arg1, args.arg2]:
        if arg:
            if arg.endswith('.csv'):

                if dataset_path:
                    print("Please provide only one dataset file.")
                    sys.exit(1)
                if not os.path.isfile(arg):
                    print(f"The file '{arg}' does not exist.")
                    sys.exit(1)
                dataset_path = arg
            else:
                if model_name:
                    print("Please provide only one model directory.")
                    sys.exit(1)
                if not os.path.isdir(arg):
                    print(f"The directory '{arg}' does not exist.")
                    sys.exit(1)
                model_name = arg

    model = load_model(model_name) if model_name else load_model()
    data = create_dataset(dataset_path) if dataset_path else create_dataset()
    general_loss = model.Optimizer.Loss.loss(model.feedforward(data.x), data.y)
    sensitivity, specificity, precision, f1 = modelGrapher.calculate_and_display_metrics(model, data.x, data.y)

    nb_pred_erreur = modelGrapher.Grapher.plot_predict(model, data, general_loss, sensitivity, specificity, precision, f1)

    print(f"nb total d'erreur = {nb_pred_erreur}")
    print(f'General Loss: {general_loss :.4f}')
    print(f'{sensitivity = :.3f}, {specificity = :.3f}, {precision = :.3f}, {f1 = :.3f}\n')