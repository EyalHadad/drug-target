import pandas as pd
import os
from src.data.data_downloader.connection_object import DataConnector
import itertools
from collections import Counter
from constants import *


def create_dict(f_path, c_name, dict_size):
    data = pd.read_csv(f_path)
    splited_data = [x.split(":") for x in data[c_name].astype(str).to_list()]
    flaten_data = list(itertools.chain(*splited_data))
    data_counter = Counter(flaten_data)
    percentage_tuples = [(i, data_counter[i] / len(flaten_data) * 100.0) for i in data_counter]
    percentage_tuples.sort(key=lambda tup: tup[1], reverse=True)
    dict_list = [(v[0], k) for k, v in enumerate(percentage_tuples[:dict_size - 1], start=1)]
    dict_list.append(('no value', dict_size))
    return pd.DataFrame(dict_list)


def create_target_dicts():
    print("---creating dictionaries---")
    print("creating keywords_dict")
    keywords_dict = create_dict(os.path.join(EXTERNAL_TRAIN_PATH, "protein_keywords.csv"), 'keywords', 400)
    keywords_dict.to_csv(os.path.join(INTERIM_PATH, 'keywords_dict.csv'), index=False, header=False)
    print("creating taxon_dict")
    taxon_dict = create_dict(os.path.join(EXTERNAL_TRAIN_PATH, "protein_taxon.csv"), 'taxon', 1000)
    taxon_dict.to_csv(os.path.join(INTERIM_PATH, 'taxon_dict.csv'), index=False, header=False)
    print("---dictionaries created successfully---")


def download_targets_data(is_train):
    dir_path = os.path.join(EXTERNAL_PATH, is_train)
    print("current path:\n", os.path.dirname(os.path.abspath(__file__)))
    dw = DataConnector(DATABASE_USERNAME, DATABASE_PASSWORD)
    dw.connect()
    dw.get_table(schema="uniport", table="gene", dst_file_path=os.path.join(dir_path, "protein_gene.csv"),
                 headers=['protein', 'gene'])
    dw.get_table(schema="uniport", table="keywords", dst_file_path=os.path.join(dir_path, "protein_keywords.csv"),
                 headers=['protein', 'keywords'])
    dw.get_table(schema="uniport", table="taxonomic_lineage", dst_file_path=os.path.join(dir_path, "protein_taxon.csv"),
                 headers=['protein', 'taxon'])
    dw.disconnect()
