from src.models.training.vanilla_models_configurations import *
from tensorflow import keras
import pandas as pd
import numpy as np
from src.models.training.training_handler import *
import os
from constants import *


class EvaluationObj:
    model_configuration_list = None
    x = None
    y = None
    ids = None
    is_target = None
    dest_name = None
    model_name = None
    pred_result = None

    def __init__(self):
        self.model_configuration_list = {"model3": train_model3, "model4": train_model4, "model5": train_model5,
                                         "model6": train_model6}

    def load_and_preprocessing(self, dest_id="", is_target=False):
        d_path = os.path.join(PROCESSED_EVALUATION_PATH, "eval_{0}.csv".format(dest_id))
        self.dest_name = dest_id
        self.is_target = is_target
        data = pd.read_csv(d_path, index_col=False)
        print("---Train data was loaded---\n")
        x = data.drop(['drugBank_id', 'gene', 'protein', 'label'], axis=1)
        x = np.asarray(x).astype('float32')
        self.y = data['label']
        print("---Finished load and preprocessing data---\n")
        ids = data[['drugBank_id', 'gene']]
        self.x = x
        self.ids = ids

    def use_model(self, _model_name):
        print("Loading model")
        self.model_name = _model_name
        if self.is_target:
            m_path = os.path.join(MODELS_OBJECTS_PATH, "{0}_{1}.h5".format(self.model_name, self.dest_name))
        else:
            m_path = os.path.join(MODELS_OBJECTS_PATH, "%s.h5" % self.model_name)
        model = keras.models.load_model(m_path)
        print("Start prediction")
        predict_res = model.predict(self.x)
        df_pred = pd.DataFrame(predict_res, columns=['prediction'])
        print("Save results")
        self.pred_result = pd.concat([self.ids, df_pred], axis=1)
        if self.is_target:
            self.add_col_names()
            self.add_belongs_to_training()
            self.known_cancer_drug()

        self.pred_result.sort_values(by=['prediction'], ascending=False, inplace=True)
        self.pred_result.to_csv(os.path.join(MODELS_PREDICTION_PATH, "pred_{0}_{1}.csv".format(self.model_name, self.dest_name)),index=False)
        calculate_auc_aupr(self.y, df_pred, self.model_name)

    def add_col_names(self):
        # TODO change drug name to test version
        drug_names = pd.read_csv(os.path.join(EXTERNAL_TRAIN_PATH, "drug_name.csv"))
        id_name_dict = pd.Series(drug_names.name.values, index=drug_names.drugBank_id).to_dict()
        self.pred_result['name'] = self.pred_result.apply(lambda row: id_name_dict[row['drugBank_id']] if row['drugBank_id'] in id_name_dict else 'None', axis=1)

    def add_belongs_to_training(self):
        train_data = pd.read_csv(os.path.join(PROCESSED_TRAIN_PATH, "train_{0}.csv".format(self.dest_name)))
        labels = set(train_data[train_data['label'] == 1]['drugBank_id'])
        self.pred_result['belongs_to_train'] = self.pred_result.apply(lambda row: 1 if row['drugBank_id'] in labels else 0, axis=1)

    def known_cancer_drug(self):
        known_cancer_drugs = pd.read_csv(os.path.join(os.path.join(RAW_PATH, "cancer_drugs.csv")))
        id_name_dict = pd.Series(known_cancer_drugs.name.values, index=known_cancer_drugs.drugBank_id).to_dict()
        self.pred_result['cancer_drug'] = self.pred_result.apply(lambda row: 1 if row['drugBank_id'] in id_name_dict else 0, axis=1)
