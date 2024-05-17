""" create earthquake vs. non-earthquake data 70-30"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

data_dir = "./data/CrisisNLP_labeled_data_crowdflower/"


# earthquake: 4 non: 7
non_folders = ['2014_Pakistan_floods', '2014_India_floods', '2015_Cyclone_Pam_en', '2014_Hurricane_Odile_Mexico_en', '2014_Philippines_Typhoon_Hagupit_en', '2014_Middle_East_Respiratory_Syndrome_en', '2014_ebola_cf']

eq_folders = ['2014_Chile_Earthquake_en', '2015_Nepal_Earthquake_en', '2013_Pakistan_eq', '2014_California_Earthquake']

non_dfs = [] 
for sub_folder in non_folders:
    if ".tsv" in os.listdir(data_dir + sub_folder)[0]:
        df = pd.read_csv(data_dir + sub_folder + '/' + os.listdir(data_dir + sub_folder)[0], sep="\t")
    else:
        df = pd.read_csv(data_dir + sub_folder + '/' + os.listdir(data_dir + sub_folder)[1], sep="\t")
    # print(df[~df.tweet_text.str.contains("earthquake", case=False)])
    non_dfs.append(df[~df.tweet_text.str.contains("earthquake", case=False)])
    # non_dfs.append(df)

eq_dfs = []
for sub_folder in eq_folders:
    if ".tsv" in os.listdir(data_dir + sub_folder)[0]:
        df = pd.read_csv(data_dir + sub_folder + '/' + os.listdir(data_dir + sub_folder)[0], sep="\t")
    else:
        df = pd.read_csv(data_dir + sub_folder + '/' + os.listdir(data_dir + sub_folder)[1], sep="\t")
    
    eq_dfs.append(df[df["label"]!="not_related_or_irrelevant"])
    non_dfs.append(df[df["label"]=="not_related_or_irrelevant"])
        
non_df = pd.concat(non_dfs)
eq_df = pd.concat(eq_dfs)

non_df["label"] = "not_eq"
eq_df["label"] = "eq"

imbalanced_df = pd.concat([non_df, eq_df])
imbalanced_df = imbalanced_df.sample(frac=1)

imbalanced_train, imbalanced_test = train_test_split(imbalanced_df, test_size=0.2)

non_df = non_df.sample(frac=0.586719)
eq_df = eq_df.sample(frac=1)

balanced_df = pd.concat([non_df, eq_df])
balanced_df = balanced_df.sample(frac=1)

balanced_train, balanced_test = train_test_split(balanced_df, test_size=0.2)


imbalanced_train.to_csv(data_dir+"event_imbalanced/train.csv", index=False)
imbalanced_test.to_csv(data_dir+"event_imbalanced/test.csv", index=False)

balanced_train.to_csv(data_dir+"event_balanced/train.csv", index=False)
balanced_test.to_csv(data_dir+"event_balanced/test.csv", index=False)
