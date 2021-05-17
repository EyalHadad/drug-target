import os
from src.data_downloader.download_drugs import download_drugs_data
from src.data_downloader.download_targets import download_targets_data
from src.data_migration.drug_bank_features import get_drug_modalities_data
from src.data_readers.create_train_data import create_training_data
from src.training.train_vanilla_modle import load_and_preprocessing,train_model


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # download_targets_data("train")
    # download_drugs_data("train","5.1.6")
    raw_data_files_path = "../../raw_data/for_train"
    # get_drug_modalities_data("5.1.6", raw_data_files_path)
    # create_training_data(raw_data_files_path)

    # _x, _y, _data_shape = load_and_preprocessing()
    # train_model("4", _x, _y, _data_shape)

    download_targets_data("test")
    # download_drugs_data("test","5.1.8")
    # get_drug_modalities_data("5.1.8", "../../raw_data/for_test")