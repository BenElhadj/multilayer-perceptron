import numpy as np
from dataset import create_dataset, KFoldIterator
from loss import is_overfitting
from collections import deque 
from loss import NAGOptimizer
import modelGrapher
import argparse
import sys
import os

def train_model(dataset_path='data.csv', model_name='model'):
    # sourcery skip: avoid-builtin-shadow
    if os.path.isdir(model_name):
        confirm = input(f"Be careful '{model_name}' directory exists.\nThe data inside will be overwritten!!!\n\n\t\tDo you want to continue? (y/n)\n")
        if confirm.lower() == 'n':
            sys.exit(0)
    if not os.path.isfile(dataset_path):
        print(f"The file '{dataset_path}' does not exist.\nHaving a dataset file is required to train the model!!!")
        sys.exit(1)
    data = create_dataset(y_categorical=True, path=dataset_path)
    kfold_iterator = KFoldIterator(data.x, data.y, 5)
    optimizer = NAGOptimizer(learning_rate=0.03, momentum=0.7)
    model = modelGrapher.Model(sizes=[data.x.shape[1], 128, 64, data.y.shape[1]], 
                               activations=['sigmoid', 'sigmoid', 'softmax'], optimizer=optimizer)
    train_dataset, test_dataset = next(kfold_iterator)
    losses = deque(maxlen=5)
    y_pred = []
    iter = 500

    for i in range(iter):
        model.train(train_dataset, 27)
        model.grapher.calculate_metrics(model, train_dataset, test_dataset, data)
        model.grapher.print_metrics(iter)

        losses.append(model.grapher.val_loss)
        y_pred.append(model.feedforward(data.x))
        if (is_overfitting(losses)):
            print(f'\nEarly Stopping in {i+2} epoch ===> Training model overfitting!!!')
            break

    general_loss = model.Optimizer.Loss.loss(model.feedforward(data.x), data.y)
    print(f'General Loss: {general_loss :.4f}')

    modelGrapher.calculate_and_display_metrics(model, data.x, data.y)
    model.save(model_name)
    data.save_norm(model_name)
    y_pred = np.array(y_pred)
    y_true = np.array(data.y)
    model.grapher.plot_metrics(y_pred ,y_true)
    
if __name__ == '__main__':
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
                dataset_path = arg
            else:
                if model_name:
                    print("Please provide only one template directory name.")
                    sys.exit(1)
                model_name = arg

    train_model(dataset_path or 'data.csv', model_name or 'model')
    create_dataset(dataset_path or None)