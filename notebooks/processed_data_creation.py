from src.data.data_creation.create_processed_data import *

if __name__ == '__main__':
    process_cancer_drugs()
    # creating train and prediction data for all drugs&targets
    # create_all_training_data()
    # create train data for specific drug
    # create_drug_data('db03419')

    # create train and prediction data for specific target
    # target_names = ['ifng', 'kat5', 'tyms', 'dhfr', 'tf', 'pdcd1', 'a2m']
    # for tar in target_names:
    #     create_target_data(tar, for_train=False)
    #     create_target_data(tar, for_train=True)
