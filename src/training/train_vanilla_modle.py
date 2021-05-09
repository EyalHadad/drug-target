import pandas as pd
from sklearn.model_selection import train_test_split
from src.training.vanilla_models_configurations import train_model3,train_model4,train_model5,train_model6
from src.training.training_handler import plot_model_configuration, save_model_and_results
from pickle import load
import numpy as np
import os

def train_model(opt,x,y,data_shape,target_data=None):
    print("---Configure network---\n")
    model = None
    if opt == "3":
        model = train_model3(data_shape)
    elif opt=="4":
        model = train_model4(data_shape)
    elif opt == "5":
        model = train_model5(data_shape)
    elif opt == "6":
        model = train_model6(data_shape)
    print("---Save network configuration---\n")
    plot_model_configuration(model, opt)
    print("---Training---\n")
    history = model.fit(x, y, batch_size=256, epochs=25, verbose=2, validation_split=0.2)
    if target_data is None:
        model_name = opt+"_reg"
    else:
        model_name = target_data+"_"+opt + "_reg"
    save_model_and_results(model, history,model_name)


def load_and_preprocessing(target_data=None):
    if target_data is None:
        data = pd.read_csv(r"../../data/train.csv", index_col=False)
    else:
        data = pd.read_csv(os.path.join(r"../../data","train_{0}.csv".format(target_data)), index_col=False)
    print("---Train data was loaded---\n")
    lbe = load(open(r"../../raw_data/for_train/dicts/weight_scalar.pkl", 'rb'))
    data['weight'] = lbe.transform(data['weight'].values.reshape(-1, 1))
    print("---Transformed drug weight feature---\n")
    print("training data shape:", data.shape)
    train, test = train_test_split(data, test_size=0.2)
    x = train.drop(['drugBank_id','gene','protein','label'], axis=1)
    x = np.asarray(x).astype('float32')
    y = train['label']
    print("---Finished load and preprocessing data---\n")
    return x,y,x.shape[1]


if __name__ == '__main__':
    _target_data = "tyms"
    _x, _y, _data_shape = load_and_preprocessing(target_data=_target_data)
    train_model("4",_x, _y, _data_shape,target_data=_target_data)
