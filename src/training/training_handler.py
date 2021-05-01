import pandas as pd
from tensorflow.python.keras.models import save_model
import tensorflow as tf
import os
from matplotlib import pyplot as plt


def show_results(history,opt="0"):
    base_path = r"../../output"

    data = pd.DataFrame(history.history)
    data.to_csv(os.path.join(base_path,opt + "_history.csv"), index=False)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Accuracy ' + opt)
    plt.legend()  # for the two lines, can get parameters such as loc='upper left' for locating the lines "menu"

    plt.savefig(os.path.join(base_path, 'acc_' + opt +'.png'))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Loss ' + opt)
    plt.legend()
    plt.savefig(os.path.join(base_path, 'loss_' + opt +'.png'))



def plot_model_configuration(model, opt):
    plot_path = os.path.join("../../output",opt + 'reg.png')
    tf.keras.utils.plot_model(model, to_file=plot_path, show_shapes=True)


def save_model_and_results(model, history, opt):
    model_path = os.path.join("../../output", opt + '_model.h5')
    print("---Save model---")
    save_model(model, model_path)
    print("---Save history and plot---")
    show_results(history, opt + "_reg")