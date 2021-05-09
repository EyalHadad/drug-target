from tensorflow import keras
from pickle import load
import pandas as pd
import numpy as np
import os




def load_and_preprocessing(drug_id,is_target):
    d_path = os.path.join(r"../../data",drug_id+".csv")
    dicts_path = "../../raw_data/for_train/dicts"
    data = pd.read_csv(d_path, index_col=False)
    print("---Train data was loaded---\n")
    lbe = load(open(os.path.join(dicts_path,"weight_scalar.pkl"), 'rb'))
    data['weight'] = lbe.transform(data['weight'].values.reshape(-1, 1))
    print("---Transformed drug weight feature---\n")
    print(data.info)
    x = data.drop(['drugBank_id','gene','protein'], axis=1)
    x = np.asarray(x).astype('float32')
    print("---Finished load and preprocessing data---\n")
    if is_target:
        ids = data['drugBank_id']
    else:
        ids = data['gene']
    return x, ids


def add_col_names(to_save):
    drug_names = pd.read_csv(r"../../raw_data/for_train/drug_name.csv")
    id_name_dict = pd.Series(drug_names.name.values, index=drug_names.drugBank_id).to_dict()
    to_save['name'] = to_save.apply(lambda row: id_name_dict[row['drugBank_id']] if row['drugBank_id'] in id_name_dict else 'None',axis=1)
    return to_save


def add_belongs_to_training(to_save, drug_target_id):
    train_data = pd.read_csv(os.path.join(r"../../data", "train_{0}.csv".format(drug_target_id)))
    labels = set(train_data[train_data['label'] == 1]['drugBank_id'])
    to_save['belongs_to_train'] = to_save.apply(lambda row: 1 if row['drugBank_id'] in labels else 0, axis=1)
    return to_save


def use_model(_x, to_predict, drug_target_id, classifier, specific_target_model = False):
    print("Loading model")
    if specific_target_model:
        m_path = os.path.join(r"../../output", "{0}_{1}_model.h5".format(drug_target_id,classifier))
    else:
        m_path = os.path.join(r"../../output", "%s_model.h5" % classifier)
    model = keras.models.load_model(m_path)
    print("Start predicting")
    predict_res = model.predict(_x)
    df_pred = pd.DataFrame(predict_res, columns=['prediction'])
    print("Save results")
    to_save = pd.concat([to_predict, df_pred], axis=1)
    if _is_target:
        to_save = add_col_names(to_save)
        to_save = add_belongs_to_training(to_save,drug_target_id)

    to_save.sort_values(by=['prediction'],ascending=False,inplace=True)
    to_save.to_csv(os.path.join(r"../../output", drug_target_id + "_prediction.csv"), index=False)


if __name__ == '__main__':
    _is_target=True
    target_names = ['ifng', 'kat5', 'tyms', 'dhfr', 'tf', 'pdcd1', 'a2m']
    for to_predict in target_names:
        _x, _ids = load_and_preprocessing(to_predict, _is_target)
        use_model(_x, _ids, to_predict, "4_reg", specific_target_model=True)



    
