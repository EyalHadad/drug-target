import os
import pandas as pd


def avg_p(x,y):
    s = x+y
    return s/2

def top_common_targets(drug_id_1,drug_id_2):
    print("Loading two data sets")
    d_1 = pd.read_csv(os.path.join(r"../../output", drug_id_1 + "_prediction.csv"))
    d_2 = pd.read_csv(os.path.join(r"../../output", drug_id_2 + "_prediction.csv"))
    comb = pd.merge(left=d_1, right=d_2, on="gene")
    comb['avg_prediction'] = comb.apply(lambda row: avg_p(row['prediction_x'],row['prediction_y']),axis=1)
    comb.sort_values(by='avg_prediction',ascending=False,inplace=True)
    print("Writing results to csv")
    comb.to_csv(os.path.join(r"../../output", drug_id_1 +"_" + drug_id_2 +"_top_targets.csv"), index=False)
    print("CSV have benn created")


if __name__ == '__main__':
    top_common_targets('DB00822','DB00641')
    i=9