import pandas as pd
import os
from sklearn.utils import shuffle
from src.data_readers.create_train_data import create_targets_dataset, create_drugs_dataset


def create_target_predicting_data(dir_path, target_id=" ",with_label=False,neg_pos_ratio = 1):
    print("---Create Target Predicting data---\n")
    all_drugs = create_drugs_dataset(dir_path)
    print("---Positive data---\n")
    pos_drugs =all_drugs[all_drugs['gene']==target_id]
    pos_drugs.reset_index(inplace=True,drop=True)
    all_targets = create_targets_dataset(dir_path, belong_to_drugs=set(all_drugs['gene']))
    target_info = all_targets[all_targets['gene'] == target_id].iloc[0:1, :]
    n_rows_pos = pos_drugs.shape[0] - 1
    pos_target_info = target_info.append([target_info] * n_rows_pos, ignore_index=True)

    pos_train_data = pd.merge(pos_drugs, pos_target_info, on='gene', how='left')
    pos_train_data.drop_duplicates(subset=['drugBank_id'], keep='first', inplace=True)

    print("Positive data shape:",pos_train_data.shape)
    print("---Negative data---\n")

    neg_drugs = all_drugs[~all_drugs['drugBank_id'].isin(pos_drugs['drugBank_id'])]
    neg_drugs.drop_duplicates(subset=['drugBank_id'], keep='first', inplace=True)
    if with_label:
        n_rows_neg = (n_rows_pos+1) * neg_pos_ratio
    else:
        n_rows_neg = neg_drugs.shape[0]

    neg_target_info = target_info.append([target_info] * n_rows_neg, ignore_index=True)
    neg_target_info.drop(['gene'], axis=1, inplace=True)
    neg_drugs = neg_drugs.sample(n=neg_target_info.shape[0]-1)
    neg_drugs.reset_index(inplace=True,drop=True)
    neg_train_data = pd.concat([neg_drugs, neg_target_info], axis=1)
    print("Negative data shape:", pos_train_data.shape)

    dest_file_name = os.path.join(r"../../data", "{0}.csv".format(target_id))
    if with_label:
        pos_train_data.insert(loc=0, column='label', value=1)
        neg_train_data.insert(loc=0, column='label', value=0)
        dest_file_name = os.path.join(r"../../data", "train_{0}.csv".format(target_id))

    train_data = pos_train_data.append(neg_train_data)
    train_data.dropna(inplace=True)
    print("Total data shape:", train_data.shape)
    train_data = shuffle(train_data)

    train_data.to_csv(dest_file_name, index=False)
    print("---",target_id,"data was created---\n")


if __name__ == '__main__':
    raw_data_files_path = "../../raw_data/for_train"
    target_names = ['ifng','kat5', 'tyms','dhfr','tf','pdcd1','a2m']
    for tar in target_names:
        create_target_predicting_data(raw_data_files_path, tar,with_label=False)
        create_target_predicting_data(raw_data_files_path, tar,with_label=True)
    i=9