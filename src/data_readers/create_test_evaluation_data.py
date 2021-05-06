from src.data_readers.create_data_handler import create_drugs_dataset, create_targets_dataset


def create_test_evaluation_data(dir_path):
    print("---Create Training data---\n")
    all_drugs = create_drugs_dataset(dir_path)
    all_targets = create_targets_dataset(dir_path, belong_to_drugs=set(all_drugs['gene']))
    all_targets.drop(['protein'], axis=1, inplace=True)
    evaluate_data = all_drugs.assign(key=0).merge(all_targets.assign(key=0), how='left', on = 'key')
    print("\nDone cross merge with all combinations. total shape:", evaluate_data.shape)
    drug_gene_dict = dict()
    for drug_id, gene in zip(evaluate_data['drugBank_id'], evaluate_data['gene']):
        drug_gene_dict.setdefault(drug_id, []).append(gene)
    evaluate_data['label'] = evaluate_data.apply(lambda row: 1 if row['gene'] in drug_gene_dict[row['drugBank_id']] else 0, axis=1)

    evaluate_data.to_csv("../../data/evaluate_data.csv", index=False)
    print("---Evaluate_data data was created---\n")


if __name__ == '__main__':
    raw_data_files_path = "../../raw_data/for_test"
    create_test_evaluation_data(raw_data_files_path)
    i=9