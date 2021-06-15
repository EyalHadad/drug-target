import pandas as pd
import os
from sklearn.utils import shuffle
from constants import *


def transform_cat_feature(data=None, column_name=None, dest_path=None):
    print("--Converting", column_name, "into one hot encoding--")
    pandas_dict = pd.read_csv(dest_path, names=['key', 'val'], index_col=False)
    f_dict = pandas_dict.set_index('key').T.to_dict('int')['val']
    embed_data = []
    dict_size = len(f_dict)
    data[column_name] = data[column_name].str.lower()
    for x in data[column_name]:
        index_to_insert = [f_dict[k] - 1 if k in f_dict else f_dict['no value'] - 1 for k in x.split(":")]
        embed_data.append([1 if x in index_to_insert else 0 for x in range(dict_size)])

    c_names = [column_name + str(i) for i in range(len(f_dict))]
    encode_feature = pd.DataFrame(data=embed_data, columns=c_names, dtype=int)
    print("previous shape:", data.shape)
    data.drop([column_name], axis=1, inplace=True)

    data = pd.concat([data, encode_feature], axis=1).reindex(data.index)
    print("updated shape:", data.shape)
    return data


def create_drugs_dataset(d_path):
    print("\n---Reading drugs data---")
    dt = pd.read_csv(os.path.join(d_path, "drug_target.csv"))
    print("done reading drug-target data with shape:", dt.shape)
    dw = pd.read_csv(os.path.join(d_path, "drug_weight.csv"))
    print("done reading drug-weight data with shape:", dw.shape)
    dc = pd.read_csv(os.path.join(d_path, "drug_cluster_features.csv"))
    print("done reading drug-cluster data with shape:", dc.shape)
    # _all_drugs = pd.merge(dw, dt, on='drugBank_id', how='left')
    _all_drugs = pd.merge(pd.merge(dw, dt, on='drugBank_id', how='left'), dc, on='drugBank_id', how='left')
    _all_drugs.dropna(inplace=True)
    print("\nDone merging drugs data. total shape:", _all_drugs.shape)
    print("---Finish reading drugs data---\n")
    return _all_drugs


def create_targets_dataset(d_path, belong_to_drugs=None):
    print("\n---Reading targets data---")
    pg = pd.read_csv(os.path.join(d_path, "protein_gene.csv"))
    print("done reading gene data with shape:", pg.shape)
    pk = pd.read_csv(os.path.join(d_path, "protein_keywords.csv"))
    pk['keywords'] = pk['keywords'].astype(str)
    print("done reading keyword data with shape:", pk.shape)
    pt = pd.read_csv(os.path.join(d_path, "protein_taxon.csv"))
    pt['taxon'] = pt['taxon'].astype(str)
    print("done reading taxonomy data with shape:", pt.shape)
    _all_targets = pd.merge(pd.merge(pg, pt, on='protein', how='left'), pk, on='protein', how='left')
    _all_targets.dropna(inplace=True)
    print("\nDone merging targets data")
    if belong_to_drugs is not None:
        _all_targets = _all_targets[_all_targets['gene'].isin(belong_to_drugs)]
        # custom_d = _all_targets[_all_targets['protein'].str.contains('HUMAN')]
        _all_targets.drop_duplicates(subset=['gene'], keep='first', inplace=True)
        print("Remove all targets without known drugs. total shape:", _all_targets.shape)

    _all_targets.reset_index(inplace=True, drop=True)
    _all_targets = transform_cat_feature(_all_targets, 'keywords', os.path.join(INTERIM_PATH, "keywords_dict.csv"))
    _all_targets = transform_cat_feature(_all_targets, 'taxon', os.path.join(INTERIM_PATH, "taxon_dict.csv"))
    print("---Finish reading targets data. final shape:", _all_targets.shape)

    return _all_targets


def create_neg_data(_all_drugs, _all_targets, n_iter=5):
    print("\n---Create negative data---")
    neg_data_list = []
    for i in range(n_iter):
        shuffle_drugs = shuffle(_all_drugs)
        shuffle_drugs.drop(['gene'], axis=1, inplace=True)
        shuffle_targets = shuffle(_all_targets)
        tmp_data = pd.concat([shuffle_drugs, shuffle_targets], axis=1)
        tmp_data.dropna(inplace=True)
        neg_data_list.append(tmp_data)
        print("generate random data with shape:", tmp_data.shape)
    neg_data = pd.concat(neg_data_list)
    return neg_data
