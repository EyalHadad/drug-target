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


def use_model(_x, _gene, drug_target_id, classifier, specific_target_model = False):
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
    to_save = pd.concat([_gene, df_pred], axis=1)
    to_save.sort_values(by=['prediction'],ascending=False,inplace=True)
    to_save.to_csv(os.path.join(r"../../output", drug_target_id + "_prediction.csv"), index=False)


if __name__ == '__main__':
    _is_target=True
    to_predict = "tyms"
    _x, _ids = load_and_preprocessing(to_predict, _is_target)
    use_model(_x, _ids, to_predict, "4_reg",specific_target_model=True)
    
