
import pandas as pd

if __name__ == '__main__':

    target_names = ['ifng', 'kat5', 'tyms', 'dhfr', 'tf', 'pdcd1', 'a2m']
    for tar in target_names:
        dir_path = r"../../output/{0}_prediction.csv".format(tar)
        data = pd.read_csv(dir_path,nrows=50)
        data.to_csv(r"../../output/short_{0}_prediction.csv".format(tar),index=False)