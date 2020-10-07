import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def label_encode_data(df):
    labelencoder = LabelEncoder()
    for col in df.columns:
        df[col] = labelencoder.fit_transform(df[col])
    return df


def one_hot_encode_data(df):
    for i in df.columns:
        dummies = pd.get_dummies(df[i], drop_first=True, prefix=i.split('_')[0])
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[i])
    df = df.rename(columns={'class_p': 'class'})
    return df


def train_test_split(df, split_ratios=None):
    if split_ratios is None:
        split_ratios = [0.6, 0.2, 0.2]
    return np.split(df.sample(frac=1), [int(split_ratios[0] * len(df)),
                                        int((split_ratios[0] + split_ratios[1]) * len(df))])
