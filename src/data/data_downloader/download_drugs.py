import pandas as pd
import os
from src.data.data_downloader.connection_object import DataConnector
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
from constants import *
from src.data.data_feature_extraction.drug_bank_features import get_drug_modalities_data


def create_minmax_scalar(dir_path, c_name):
    scaler = MinMaxScaler()
    data = pd.read_csv(os.path.join(dir_path, "drug_weight.csv"))
    scaler.fit(data[[c_name]])
    print("transform data using MinMax scalar")
    data[c_name] = scaler.transform(data[[c_name]])
    dump(scaler, open(os.path.join(INTERIM_PATH, c_name + '_scalar.pkl'), 'wb'))
    print("MinMax scalar saved successfully")
    data.to_csv(os.path.join(dir_path, "drug_weight.csv"), index=False)
    print(c_name, "data scaled and saved successfully")


def download_drugs_data(is_train, version):
    print("\n---Downloading version", version, "---")
    dir_path = os.path.join(EXTERNAL_PATH, is_train)
    dw = DataConnector(DATABASE_USERNAME, DATABASE_PASSWORD)
    dw.connect()
    table_name = "Target_" + version
    dw.get_table(schema="DrugBank", table=table_name, dst_file_path=os.path.join(dir_path, "drug_target.csv"),
                 headers=['drugBank_id', 'gene'])
    table_name = "Molecular_weight_" + version
    dw.get_table(schema="DrugBank", table=table_name, dst_file_path=os.path.join(dir_path, "drug_weight.csv"),
                 headers=['drugBank_id', 'weight'])
    if is_train == "train":
        # TODO add drug_name table to test version
        table_name = "drug_Name_" + version
        dw.get_table(schema="DrugBank", table=table_name, dst_file_path=os.path.join(dir_path, "drug_name.csv"),
                     headers=['drugBank_id', 'name'])
        print("creating MinMax weight scalar")
        create_minmax_scalar(dir_path, 'weight')
    dw.disconnect()

