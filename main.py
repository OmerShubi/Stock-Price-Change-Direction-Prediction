"""
PART 1

1. preprocess data - import and clean
    drop OpenInt
    create binary label and drop close

    for structure models:
        groupby week - group all days of week into one row, keep only full weeks (have 5 days)

Perceptron:
    2. Split train test, and X Y -
        80 - 20 (first 80% of days) # TODO make sure not split random
    3. Train perceptron on train
    4. predict perceptron on test
    5. show accuracy and graph

MEMM:
    Create list of lists (day features for each week)
    Add features - TODO
    Train Model
    Test Model
    show accuracy and graph

LSTM:
    Train Model
    Test Model
    show accuracy and graph


Compare results
"""

import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
import random
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable

TEST_SIZE = 0.2
SHUFFLE_TRAIN_TEST = False
FEATURES = ['Open', 'High', 'Low', 'Volume']


def pre_process(input_df):
    print("pre_processing")
    df = input_df.copy()
    # check OpenInt values
    # if df[df.OpenInt > 0].empty: print("all OpenInt values are zero")
    # remove OpenInt column
    df.drop("OpenInt", axis=1, inplace=True)
    # remove High, low columns
    # df.drop(["High",'Low'],axis=1, inplace=True)
    # add direction column : 1 if Close price >= Open price else 0
    df.loc[df.Close >= df.Open, 'direction'] = 1
    df.loc[df.Close < df.Open, 'direction'] = 0
    df.drop(["Close"], axis=1, inplace=True)
    # check days range data
    print(f'Data from day {min(df.Date)} to day {max(df.Date)}')
    print('Num of days ', len(df))
    return df


def preprocess_to_week(input_df):
    df = input_df.copy()
    # add day of week
    df['weekDay'] = df.Date.dt.weekday

    # give index to each week
    week = set()
    week_index = 0
    for index, row in df.iterrows():
        if row.weekDay not in week and all([x < row.weekDay for x in week]):
            week.add(row.weekDay)
            df.at[index, 'weekNum'] = week_index
        else:
            week_index = week_index + 1
            week = set()
            week.add(row.weekDay)
            df.at[index, 'weekNum'] = week_index

    # if week is not full (doesn't have 5 days) --> drop week
    for index in range(int(max(df.weekNum)) + 1):
        week_days = df[df.weekNum == index]
        if len(week_days) != 5:
            df.drop(df[df.weekNum == index].index, inplace=True)

    # reindex
    df.reset_index(inplace=True, drop=True)
    week_days_index = 0
    for index, row in df.iterrows():
        df.at[index, 'weekNum'] = week_days_index
        if (index + 1) % 5 == 0:  # and index != 0:
            week_days_index += 1

    df.drop(["Date"], axis=1, inplace=True)

    num_of_weeks = max(df.weekNum)
    res = []
    features_len = len(FEATURES)
    for week_num in range(int(num_of_weeks)):
        week_feature_list = []
        week_label_list = []
        week = df[df['weekNum'] == week_num]
        for _, day in week.iterrows():
            week_feature_list.append(day[0:features_len].to_numpy())
            week_label_list.append(day[-1])
        res.append((torch.tensor(week_feature_list, dtype=torch.long)
                    , torch.tensor(week_label_list, dtype=torch.long)))  # TODO require grad?

    # group all days of week into one row
    # df['Open'] = df.groupby('weekNum')['Open'].apply(list)
    # df['Volume'] = df.groupby('weekNum')['Volume'].apply(list)
    # df['High'] = df.groupby('weekNum')['High'].apply(list)
    # df['Low'] = df.groupby('weekNum')['Low'].apply(list)
    # df['direction'] = df.groupby('weekNum')['direction'].apply(list)
    #
    # cleanup mess
    # df.drop(["weekDay", 'weekNum'], axis=1, inplace=True)
    # df.reset_index(inplace=True, drop=True)
    # df = df.head(int(num_of_weeks))

    # print(f'num_of_weeks={num_of_weeks}')
    return res


def load_data():
    try:
        df_day = pd.read_pickle('df_da1y.pkl')
        df_week = pd.read_pickle('df_week.pkl')
        print('loaded data')
    except:
        print('created data')
        df = pd.read_csv('ibm.us.txt', parse_dates=['Date'], index_col=['index'])
        df_day = pre_process(df)
        df_week = preprocess_to_week(df_day)
    return df_day, df_week


def perceptron_phase(df_day):
    X = df_day[FEATURES]
    y = df_day['direction']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TEST_SIZE,
                                                        random_state=42,
                                                        shuffle=SHUFFLE_TRAIN_TEST)

    clf = Perceptron(random_state=0)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


def main():
    df_day, df_week = load_data()

    perceptron_phase(df_day)

    ######## LSTM
    # df_week
    pass

if __name__ == '__main__':
    main()
