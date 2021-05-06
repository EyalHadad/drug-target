import pandas as pd
from sklearn.utils import shuffle
from src.data_readers.create_data_handler import create_drugs_dataset, create_targets_dataset


def create_neg_data(_all_drugs,_all_targets,n_iter=5):
    print("\n---Create negative data---")
    neg_data_list = []
    for i in range(n_iter):
        shuffle_drugs = shuffle(_all_drugs)
        shuffle_drugs.drop(['gene'], axis=1, inplace=True)
        shuffle_targets = shuffle(_all_targets)
        tmp_data = pd.concat([shuffle_drugs, shuffle_targets], axis=1)
        tmp_data.dropna(inplace=True)
        neg_data_list.append(tmp_data)
        print("generate random data with shape:",tmp_data.shape)

    neg_data = pd.concat(neg_data_list)
    return neg_data


def create_training_data(dir_path):
    print("---Create Training data---\n")
    all_drugs = create_drugs_dataset(dir_path)
    all_targets = create_targets_dataset(dir_path, belong_to_drugs=set(all_drugs['gene']))
    neg_data = create_neg_data(all_drugs,all_targets)
    print("Done merging drugs with random targets (negative examples).shape:", neg_data.shape)
    train_data = pd.merge(all_drugs,all_targets,on='gene',how='left')
    train_data.dropna(inplace=True)
    print("\nDone merging all drugs with targets (positive examples). total shape:", train_data.shape)
    total = train_data.append(neg_data)
    drug_gene_dict = dict()
    for drug_id, gene in zip(all_drugs['drugBank_id'], all_drugs['gene']):
        drug_gene_dict.setdefault(drug_id, []).append(gene)
    total['label'] = total.apply(lambda row: 1 if row['gene'] in drug_gene_dict[row['drugBank_id']] else 0, axis=1)
    total = shuffle(total)
    total.to_csv("../../data/train.csv",index=False)
    print("---Training data was created---\n")


if __name__ == '__main__':
    raw_data_files_path = "../../raw_data/for_train"
    create_training_data(raw_data_files_path)
    i=9