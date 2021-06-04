import pandas as pd
from sklearn import metrics
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
    x_test = None
    y_test = None
    train_shape = None
    test_shape = None
    dest_name = None
    model_name = None
    model = None

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
        self.x_test = test.drop(['drugBank_id', 'gene', 'protein', 'label'], axis=1)
        self.x_test = np.asarray(self.x_test).astype('float32')
        self.y_test = test['label']
        self.train_shape = x.shape[1]
        self.test_shape = self.x_test.shape[1]
        print("---Finished load and preprocessing data---\n")

    def train_model(self, _model_name):
        print("---Configure network---\n")
        if self.dest_name is None:
            self.model_name = _model_name
        else:
            self.model_name = _model_name + "_" + self.dest_name
        model = self.model_configuration_list[_model_name](self.train_shape)
        print("---Save network configuration---\n")
        # plot_model_configuration(model, self.model_name)
        print("---Training---\n")
        history = model.fit(self.x, self.y, batch_size=256, epochs=5, verbose=2, validation_split=0.2)
        self.model = model
        save_model_and_results(model, history, self.model_name)

    def predict_local(self, target_name=None):
        print("---Predicting local---\n")
        predict_res = self.model.predict(self.x_test)
        df_pred = pd.DataFrame(predict_res, columns=['prediction'])
        pred_result = pd.concat([self.y_test.reset_index(drop=True), df_pred], axis=1)
        pred_result.sort_values(by=['prediction'], ascending=False, inplace=True)
        pred_result.to_csv(os.path.join(MODELS_PREDICTION_PATH, "pred_{0}_local.csv".format(target_name)), index=False)
        fpr, tpr, thresholds = metrics.roc_curve(self.y_test, df_pred, pos_label=2)
        auc = metrics.auc(fpr, tpr)
        print("AUC:", auc)
        accuracy = metrics.accuracy_score(self.y_test, df_pred)
        print("ACC:", accuracy)
