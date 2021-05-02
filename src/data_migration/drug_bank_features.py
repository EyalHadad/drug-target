import random
import os
from src.data_migration.table_names import *
from src.data_migration.data_migration_handler import *
from src.data_migration.get_drug_bank_features import GetDrugBankFeaturesFromDB
from group_lasso import LogisticGroupLasso


def get_csvs(version):
    t = GetDrugBankFeaturesFromDB()
    X_unlabled, modalities_df = t.combine_features([category_table,
                                                    ATC_table_1, ATC_table_2, ATC_table_3, ATC_table_4, ATC_table_5,
                                                    # ATC_table_1_description,ATC_table_2_description,ATC_table_3_description,ATC_table_4_description,
                                                    enzyme_table, carrier_table, target_table, transporter_table,
                                                    associated_condition_table, group_table, type_table],
                                                   dense_table_name_list=[tax_table], version=version,
                                                   add_counts_to_sparse=True)  # smiles_table, mol_weight_table
    for c in list(modalities_df['Taxonomy']):
        print(c)
        X_unlabled, modalities_df = encode_and_bind(X_unlabled, c, modalities_df, 'Taxonomy')
    modalities_df = pd.DataFrame({'modality': [x for x in sorted(modalities_df.keys()) for y in modalities_df[x]],
                                  'feature': [y for x in sorted(modalities_df.keys()) for y in modalities_df[x]]})
    return X_unlabled, modalities_df


def calculate_clusters(x_unlabled, modalities_df, dir_path):
    LogisticGroupLasso.LOG_LOSSES = True
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    # Setting relevant modalities
    base_mods = ['ATC_Level_1_description', 'ATC_Level_2_description', 'ATC_Level_3_description',
                 'ATC_Level_4_description', 'ATC_Level_5', 'Associated_condition', 'Carrier', 'Category', 'Enzyme',
                 'Group', 'Target', 'Taxonomy', 'Transporter', 'Type']
    mods_domain_expert = ['Category', 'ATC_Level_3_description', 'ATC_Level_2_description', 'ATC_Level_4_description',
                          'Associated_condition', 'Taxonomy']

    # Adding our features
    X_unlabled_text = extract_text_features(
        x_unlabled[modalities_df.loc[modalities_df.modality.isin(mods_domain_expert), 'feature']])
    x_unlabled, modalities_df = add_mods(x_unlabled, X_unlabled_text, modalities_df, 'Text')
    print(X_unlabled_text.columns)
    text_mods = ['Text']
    print('done processing text')
    num_clusters = 3600  # Number of custers to create. 3600 was optimal in paper.
    clustering_mods = []
    X_cluster = get_col_clusters(
        x_unlabled[modalities_df.loc[modalities_df.modality.isin(mods_domain_expert), 'feature']], num_clusters)
    print('clusters head:')
    print(X_cluster.head())
    print(X_cluster.columns)
    mod_name = 'Clusters'
    clustering_mods.append(mod_name)
    x_unlabled, modalities_df = add_mods(x_unlabled, X_cluster, modalities_df, mod_name)
    print('done clustering')
    # Writing new data to disk
    x_unlabled=x_unlabled.apply(lambda x: x.astype(str).str.lower())
    x_unlabled = x_unlabled.filter(regex=("Cluster 3600*"))
    x_unlabled.reset_index(inplace=True)
    x_unlabled['drugBank_id'] = x_unlabled['drugBank_id'].str.lower()
    x_unlabled.to_csv(os.path.join(dir_path, 'drug_cluster_features.csv'),index=False)
    modalities_df.to_csv(os.path.join(dir_path,'modalities_w_text.csv'))




def get_drug_modalities_data(version,dir_path):
    random.seed(30)
    print("-----Get drugbank modalities data version",version,"-----")
    X_unlabled, modalities_df = get_csvs(version)
    print("-----modalities data was created-----")
    print("-----Start calculate clusters-----")
    calculate_clusters(X_unlabled, modalities_df,dir_path)
    print("---- modalities clusters was written to files-----")





