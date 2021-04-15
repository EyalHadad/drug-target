import pandas as pd
import os
from src.data_downloader.connection_object import DataConnector
from sklearn.preprocessing import MinMaxScaler
from pickle import dump


def create_minmax_scalar(dir_path, c_name):
    dict_dir = os.path.join(dir_path, 'dicts')
    if not os.path.exists(dict_dir):
        print("dictionaries folder was created")
        os.makedirs(dict_dir)
    scaler = MinMaxScaler()
    data = pd.read_csv(os.path.join(dir_path, "drug_weight.csv"))
    scaler.fit(data[[c_name]])
    print("transform data using MinMax scalar")
    data[c_name] = scaler.transform(data[[c_name]])
    dump(scaler, open(os.path.join(dict_dir,c_name+'_scaler.pkl'), 'wb'))
    print("MinMax scalar saved successfully")
    data.to_csv(os.path.join(dir_path, "drug_weight.csv"),index=False)
    print(c_name, "data scaled and saved successfully")


def download_drugs_data(is_train,version):
    print("\n---Downloading version",version,"---")
    dir_path = "../../raw_data/for_"+is_train
    dw = DataConnector(user="drugsmaster", password="pass2DRUGS!")
    dw.connect()
    table_name = "Target_"+version
    dw.get_table(schema="DrugBank",table=table_name,dst_file_path=os.path.join(dir_path,"drug_target.csv"),headers=['drugBank_id','gene'])
    table_name = "Molecular_weight_"+version
    dw.get_table(schema="DrugBank",table=table_name,dst_file_path=os.path.join(dir_path,"drug_weight.csv"),headers=['drugBank_id','weight'])
    dw.disconnect()

    if is_train=="train":
        print("creating MinMax weight scalar")
        create_minmax_scalar(dir_path, 'weight')


if __name__ == '__main__':
    download_drugs_data("train","5.1.6")
    download_drugs_data("test","5.1.8")