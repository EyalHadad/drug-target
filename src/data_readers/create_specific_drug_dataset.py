import pandas as pd
import os
from src.data_readers.create_train_data import create_targets_dataset, create_drugs_dataset


def create_drug_predicting_data(dir_path, drug_id=" "):
    print("---Create Predicting data---\n")
    all_drugs = create_drugs_dataset(dir_path)
    all_targets = create_targets_dataset(dir_path, belong_to_drugs=set(all_drugs['gene']))
    gene_col = all_targets['gene']
    all_targets = all_targets.drop(['gene'],axis=1)
    n_rows = all_targets.shape[0] -1
    drug_info = all_drugs[all_drugs['drugBank_id']==drug_id].iloc[0:1,:]
    drug_info = drug_info.append([drug_info]*n_rows,ignore_index=True)

    df_id_weight = drug_info[['drugBank_id', 'weight']]
    drug_info = drug_info.drop(['drugBank_id', 'weight','gene'],axis=1)

    print("all targets shape:",all_targets.shape)
    print("drug_info shape:",drug_info.shape)

    res = pd.concat([df_id_weight, gene_col, drug_info, all_targets], axis=1).reset_index(drop=True)

    print("predicting dataset shape:",res.shape)
    f_name = os.path.join(r"../../data",drug_id+".csv")
    res.to_csv(f_name, index=False)
    print("---drug id ",drug_id,"was created---\n")


if __name__ == '__main__':
    raw_data_files_path = "../../raw_data/for_test"
    create_drug_predicting_data(raw_data_files_path, 'db03419')
    i=9