from tensorflow import keras
from pickle import load
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os




def load_and_preprocessing(drug_id):
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
    return x,data['gene']


def use_model(_x, _gene, drug_id,classifier):
    print("Loading model")
    m_path = os.path.join(r"../../output", "%s_model.h5" % classifier)
    model = keras.models.load_model(m_path)
    print("Start predicting")
    predict_res = model.predict(_x)
    df_pred = pd.DataFrame(predict_res, columns=['prediction'])
    print("Save results")
    to_save = pd.concat([_gene, df_pred], axis=1)
    to_save.sort_values(by=['prediction'],ascending=False,inplace=True)
    to_save.to_csv(os.path.join(r"../../output",drug_id+"_prediction.csv"),index=False)


if __name__ == '__main__':
    # pred_csv = pd.read_csv(r"C:\Users\Eyal-TLV\Desktop\drug-target\data\db03419.csv",nrows=5)
    # print(pred_csv.shape)
    # train_csv = pd.read_csv(r"C:\Users\Eyal-TLV\Desktop\drug-target\data\train.csv",nrows=5)
    # print(train_csv.shape)
    _drug_id = "db03419"
    _x, gene = load_and_preprocessing(_drug_id)
    use_model(_x, gene,_drug_id,"4_reg")
    
