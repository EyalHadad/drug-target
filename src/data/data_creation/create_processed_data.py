from unittest.mock import inplace

from src.data.data_creation.create_data_handler import *


def convert_target_to_list(target, drug, dt_dict):
    if target != 'nan':
        return target.split(',')
    elif drug in dt_dict:
        return dt_dict[drug]
    else:
        return None


def process_cancer_drugs():
    print("---process cancer drugs---\n")
    drug_target = pd.read_csv(os.path.join(EXTERNAL_TEST_PATH, 'drug_target.csv'))
    drug_target = (drug_target.groupby(['drugBank_id']).agg({'gene': lambda x: x.tolist()}).reset_index())
    drug_target_dict = dict(zip(drug_target.drugBank_id, drug_target.gene))
    cancer_data = pd.read_csv(os.path.join(RAW_PATH, 'cancer_drugs.csv'),
                              usecols=['drugBank_id', 'target', 'cancer_desc'])
    cancer_data = cancer_data.apply(lambda x: x.astype(str).str.lower())
    # convert str to list and fill targets from drugbank if exists
    cancer_data['target'] = [convert_target_to_list(d_target, d_name, drug_target_dict) for d_name, d_target in
                             zip(cancer_data['drugBank_id'], cancer_data['target'])]
    cancer_data.dropna(inplace=True)
    all_drugs = create_drugs_dataset(EXTERNAL_TEST_PATH).reset_index()
    print("drugs with genes:", all_drugs.shape)
    all_drugs.drop(['index','gene'],axis=1,inplace=True)
    all_drugs = all_drugs.groupby('drugBank_id').first().reset_index()
    print("all drugs:", all_drugs.shape)
    all_drugs = all_drugs[all_drugs['drugBank_id'].isin(cancer_data['drugBank_id'])]
    print("relevant drugs:", all_drugs.shape)
    all_targets = create_targets_dataset(EXTERNAL_TEST_PATH, belong_to_drugs=set(cancer_data['target'].explode('target')))
    print("relevant targets:", all_targets.shape)
    cancer_data_explode = cancer_data.explode('target').rename(columns={"cancer_desc": "label","target":"gene"}).reset_index(drop=True)
    cancer_data_explode['label'] = cancer_data_explode['label'].replace({'false': 0, 'true': 1})
    to_save = cancer_data_explode.merge(all_drugs,left_on='drugBank_id', right_on='drugBank_id').merge(all_targets,left_on='gene', right_on='gene')
    to_save = shuffle(to_save)
    to_save.to_csv(os.path.join(PROCESSED_TRAIN_PATH, "train_cancer.csv"), index=False)
    print("---Cancer training data was created---\n")



def create_all_training_data():
    print("---Create Training data---\n")
    all_drugs = create_drugs_dataset(EXTERNAL_TRAIN_PATH)
    all_targets = create_targets_dataset(EXTERNAL_TRAIN_PATH, belong_to_drugs=set(all_drugs['gene']))
    neg_data = create_neg_data(all_drugs, all_targets)
    print("Done merging drugs with random targets (negative examples).shape:", neg_data.shape)
    train_data = pd.merge(all_drugs, all_targets, on='gene', how='left')
    train_data.dropna(inplace=True)
    print("\nDone merging all drugs with targets (positive examples). total shape:", train_data.shape)
    total = train_data.append(neg_data)
    drug_gene_dict = dict()
    for drug_id, gene in zip(all_drugs['drugBank_id'], all_drugs['gene']):
        drug_gene_dict.setdefault(drug_id, []).append(gene)
    total['label'] = total.apply(lambda row: 1 if row['gene'] in drug_gene_dict[row['drugBank_id']] else 0, axis=1)
    total = shuffle(total)
    total.to_csv(os.path.join(PROCESSED_TRAIN_PATH, "train.csv"), index=False)
    print("---Training data was created---\n")


def create_target_data(target_id=" ", for_train=False, neg_pos_ratio=1):
    print("---Create Target prediction data---\n")
    if for_train:
        dir_path = EXTERNAL_TRAIN_PATH
        dest_file_name = os.path.join(PROCESSED_TRAIN_PATH, "train_{0}.csv".format(target_id))
    else:
        dir_path = EXTERNAL_TEST_PATH
        dest_file_name = os.path.join(PROCESSED_EVALUATION_PATH, "eval_{0}.csv".format(target_id))
    all_drugs = create_drugs_dataset(dir_path)
    print("---Positive data---\n")
    pos_drugs = all_drugs[all_drugs['gene'] == target_id]
    pos_drugs.reset_index(inplace=True, drop=True)
    all_targets = create_targets_dataset(dir_path, belong_to_drugs=set(all_drugs['gene']))
    target_info = all_targets[all_targets['gene'] == target_id].iloc[0:1, :]
    n_rows_pos = pos_drugs.shape[0] - 1
    pos_target_info = target_info.append([target_info] * n_rows_pos, ignore_index=True)

    pos_train_data = pd.merge(pos_drugs, pos_target_info, on='gene', how='left')
    pos_train_data.drop_duplicates(subset=['drugBank_id'], keep='first', inplace=True)

    print("Positive data shape:", pos_train_data.shape)
    print("---Negative data---\n")

    neg_drugs = all_drugs[~all_drugs['drugBank_id'].isin(pos_drugs['drugBank_id'])]
    neg_drugs.drop_duplicates(subset=['drugBank_id'], keep='first', inplace=True)
    if for_train:
        n_rows_neg = (n_rows_pos + 1) * neg_pos_ratio
    else:
        n_rows_neg = neg_drugs.shape[0]

    neg_target_info = target_info.append([target_info] * n_rows_neg, ignore_index=True)
    neg_target_info.drop(['gene'], axis=1, inplace=True)
    neg_drugs = neg_drugs.sample(n=neg_target_info.shape[0] - 1)
    neg_drugs.reset_index(inplace=True, drop=True)
    neg_train_data = pd.concat([neg_drugs, neg_target_info], axis=1)
    print("Negative data shape:", pos_train_data.shape)

    if for_train:
        pos_train_data.insert(loc=0, column='label', value=1)
        neg_train_data.insert(loc=0, column='label', value=0)

    train_data = pos_train_data.append(neg_train_data)
    train_data.dropna(inplace=True)
    print("Total data shape:", train_data.shape)
    train_data = shuffle(train_data)

    train_data.to_csv(dest_file_name, index=False)
    print("---", target_id, "data was created---\n")


def create_drug_data(drug_id=" "):
    # TODO add option for prediction data (similar to target data)
    print("---Create prediction data---\n")
    all_drugs = create_drugs_dataset(EXTERNAL_TRAIN_PATH)
    all_targets = create_targets_dataset(EXTERNAL_TRAIN_PATH, belong_to_drugs=set(all_drugs['gene']))
    gene_col = all_targets['gene']
    all_targets = all_targets.drop(['gene'], axis=1)
    n_rows = all_targets.shape[0] - 1
    drug_info = all_drugs[all_drugs['drugBank_id'] == drug_id].iloc[0:1, :]
    drug_info = drug_info.append([drug_info] * n_rows, ignore_index=True)
    df_id_weight = drug_info[['drugBank_id', 'weight']]
    drug_info = drug_info.drop(['drugBank_id', 'weight', 'gene'], axis=1)
    print("all targets shape:", all_targets.shape)
    print("drug_info shape:", drug_info.shape)
    res = pd.concat([df_id_weight, gene_col, drug_info, all_targets], axis=1).reset_index(drop=True)
    print("prediction dataset shape:", res.shape)
    f_name = os.path.join(os.path.join(PROCESSED_TRAIN_PATH, "train_" + drug_id + ".csv"))

    res.to_csv(f_name, index=False)
    print("---drug id ", drug_id, "was created---\n")


def create_test_evaluation_data():
    print("---Create Training data---\n")
    all_drugs = create_drugs_dataset(EXTERNAL_TEST_PATH)
    all_targets = create_targets_dataset(EXTERNAL_TEST_PATH, belong_to_drugs=set(all_drugs['gene']))
    all_targets.drop(['protein'], axis=1, inplace=True)
    evaluate_data = all_drugs.assign(key=0).merge(all_targets.assign(key=0), how='left', on='key')
    print("\nDone cross merge with all combinations. total shape:", evaluate_data.shape)
    drug_gene_dict = dict()
    for drug_id, gene in zip(evaluate_data['drugBank_id'], evaluate_data['gene']):
        drug_gene_dict.setdefault(drug_id, []).append(gene)
    evaluate_data['label'] = evaluate_data.apply(
        lambda row: 1 if row['gene'] in drug_gene_dict[row['drugBank_id']] else 0, axis=1)

    evaluate_data.to_csv(os.path.join(PROCESSED_EVALUATION_PATH, "evaluate_data.csv"), index=False)
    print("---Evaluate_data data was created---\n")
