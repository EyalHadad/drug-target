from src.data.data_downloader.download_drugs import *
from src.data.data_downloader.download_targets import *

if __name__ == '__main__':
    download_evaluation_data()
    # download_targets_data("train")
    # download_drugs_data("train", "5.1.6")
    # get_drug_modalities_data("train", "5.1.6")
    # download_targets_data("test")
    # download_drugs_data("test", "5.1.8")
    # get_drug_modalities_data("test", "5.1.8")
    # create_target_dicts()
