import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.training.vanilla_models_configurations import *
from src.models.training.training_handler import plot_model_configuration, save_model_and_results
import numpy as np
import os
from constants import *


class TrainObj:
    model_configuration_list = None
    x = None
    y = None
    train_shape = None
    dest_name = None
    model_name = None

    def __init__(self):
        self.model_configuration_list = {"model3": train_model3, "model4": train_model4, "model5": train_model5,
                                         "model6": train_model6}

    def load_and_preprocessing(self, target_data=None):
        if target_data is None:
            data = pd.read_csv(os.path.join(PROCESSED_TRAIN_PATH, "train.csv"), index_col=False)
        else:
            self.dest_name = target_data
            data = pd.read_csv(os.path.join(PROCESSED_TRAIN_PATH, "train_{0}.csv".format(target_data)), index_col=False)
        print("---Train data was loaded---\n")
        print("training data shape:", data.shape)
        train, test = train_test_split(data, test_size=0.2)
        x = train.drop(['drugBank_id', 'gene', 'protein', 'label'], axis=1)
        self.x = np.asarray(x).astype('float32')
        self.y = train['label']
        self.train_shape = x.shape[1]
        print("---Finished load and preprocessing data---\n")

    def train_model(self, _model_name):
        print("---Configure network---\n")
        if self.dest_name is None:
            self.model_name = _model_name
        else:
            self.model_name = _model_name + "_" + self.dest_name
        model = self.model_configuration_list[_model_name](self.train_shape)
        print("---Save network configuration---\n")
        plot_model_configuration(model, self.model_name)
        print("---Training---\n")
        history = model.fit(self.x, self.y, batch_size=256, epochs=25, verbose=2, validation_split=0.2)
        save_model_and_results(model, history, self.model_name)
