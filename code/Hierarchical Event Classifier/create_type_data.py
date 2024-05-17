import pandas as pd
import os
from sklearn.model_selection import train_test_split

eq_folders = ['2014_Chile_Earthquake_en', '2015_Nepal_Earthquake_en', '2013_Pakistan_eq', '2014_California_Earthquake']

data_dir = "./data/CrisisNLP_labeled_data_crowdflower/"

eq_dfs = []
for sub_folder in eq_folders:
    if ".tsv" in os.listdir(data_dir + sub_folder)[0]:
        df = pd.read_csv(data_dir + sub_folder + '/' + os.listdir(data_dir + sub_folder)[0], sep="\t")
    else:
        df = pd.read_csv(data_dir + sub_folder + '/' + os.listdir(data_dir + sub_folder)[1], sep="\t")
    print(sub_folder)
    print(df["label"].value_counts())
    eq_dfs.append(df)

eq_df = pd.concat(eq_dfs)
stats_df = eq_df[eq_df["label"]=="injured_or_dead_people"]
non_df = eq_df[eq_df["label"]!="injured_or_dead_people"].sample(969)
non_df["label"] = "unrelated"

eq_df = pd.concat([stats_df, non_df])
eq_df = eq_df.sample(frac=1)

train, test = train_test_split(eq_df, test_size=0.2)

train.to_csv(data_dir+"type_balanced/train.csv", index=False)
test.to_csv(data_dir+"type_balanced/test.csv", index=False)
