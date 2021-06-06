import pandas as pd
from datetime import datetime
from tensorflow.python.keras.models import save_model
import tensorflow as tf
import os
import csv
from matplotlib import pyplot as plt
from constants import *


def show_results(history, name):
    data = pd.DataFrame(history.history)
    data.to_csv(os.path.join(MODELS_OUTPUT_PATH, name + "_history.csv"), index=False)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Accuracy ' + name)
    plt.legend()  # for the two lines, can get parameters such as loc='upper left' for locating the lines "menu"

    plt.savefig(os.path.join(MODELS_OUTPUT_PATH, 'acc_' + name + '.png'))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Loss ' + name)
    plt.legend()
    plt.savefig(os.path.join(MODELS_OUTPUT_PATH, 'loss_' + name + '.png'))


def plot_model_configuration(model, model_name):
    plot_path = os.path.join(MODELS_OUTPUT_PATH, model_name + '.png')
    tf.keras.utils.plot_model(model, to_file=plot_path, show_shapes=True)


def save_model_and_results(model, history, name):
    model_path = os.path.join(MODELS_OBJECTS_PATH, name + '.h5')
    print("---Save model---")
    save_model(model, model_path)
    print("---Save history and plot---")
    show_results(history, name)


def save_metrics(auc, aupr, model_name):
    f_path = os.path.join(MODELS_PATH, 'models_evaluation.csv')
    with open(f_path, 'a') as file:
        headers = ['Date', 'AUC', 'AUPR', 'Model']
        writer = csv.DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=headers)
        if file.tell() == 0:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow(
            {'Date': datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 'AUC': auc, 'AUPR': aupr, 'Model': model_name})
