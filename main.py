#%%
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
from sklearn.metrics import accuracy_score

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
batch_size = 10

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
    df = df[FEATURES + ['weekNum', 'direction']]
    num_of_weeks = max(df.weekNum)
    features = []
    targets = []
    features_len = len(FEATURES)
    for week_num in range(int(num_of_weeks)):
        week_feature_list = []
        week_label_list = []
        week = df[df['weekNum'] == week_num]
        for _, day in week.iterrows():
            week_feature_list.append(day[0:features_len].to_numpy())
            week_label_list.append(day[-1])
        features.append(week_feature_list)
        targets.append(week_label_list)  # TODO require grad?

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
    return features, targets


def load_data():
    try:
        df_day = pd.read_pickle('df_day.pkl')
        week_features = np.load('week_features.npy')
        week_targets = np.load('week_targets.npy')
        # df_week = pd.read_pickle('df_week.pkl')
        print('loaded data')
    except:
        print('created data')
        df = pd.read_csv('ibm.us.txt', parse_dates=['Date'], index_col=['index'])
        df_day = pre_process(df)
        week_features, week_targets = preprocess_to_week(df_day)
    return df_day, week_features, week_targets


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

class LSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super(LSTM, self).__init__()
            # Hidden dimensions
            self.hidden_dim = hidden_dim

            # Number of hidden layers
            self.num_layers = num_layers

            # Building your LSTM
            # batch_first=True causes input/output tensors to be of shape
            # (batch_dim, seq_dim, hidden_dim). with out - (seq_dim, batch_dim,, hidden_dim)
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers) # Bi True?

            # Readout layer
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):


            # One time step
            # We need to detach as we are doing truncated backpropagation through time (BPTT)
            # If we don't, we'll backprop all the way to the start even after going through another batch
            out, _ = self.lstm(x)

            # Index hidden state of last time step
            # out.size() --> 100, 28, 100
            # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
            # out = self.fc(out[:, -1, :])
            # out.size() --> 100, 10
            out = self.fc(out)
            return out

def test(model, device, test_loader):
    # model in eval mode skips Dropout etc
    model.eval()
    y_true = []
    y_pred = []

    # set the requires_grad flag to false as we are in the test mode
    with torch.no_grad():
        for i in test_loader:
            # LOAD THE DATA IN A BATCH
            data, target = i

            # moving the tensors to the configured device
            data, target = data.to(device), target.to(device)

            output = model(data.float())

            # PREDICTIONS
            pred = np.round(torch.sigmoid(output))
            target = target.float()
            y_true.extend(target.tolist())
            y_pred.extend(pred.reshape(-1).tolist())

    print("Accuracy on test set is", accuracy_score(y_true, y_pred))
    print("********************************************************")


def main():
    df_day, week_features, week_targets = load_data()
    # week_features = np.array(week_features)
    # week_targets = np.array(week_targets)
    # np.save('week_features.npy', week_features)
    # np.save('week_targets.npy', week_targets)
    # perceptron_phase(df_day)

    ######## LSTM
    X_train, X_test, y_train, y_test = train_test_split(week_features,
                                                        week_targets,
                                                        test_size=TEST_SIZE,
                                                        random_state=42,
                                                        shuffle=SHUFFLE_TRAIN_TEST)
    # make training and test sets in torch
    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    X_test = torch.from_numpy(X_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    train = torch.utils.data.TensorDataset(X_train, y_train)
    test = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=batch_size,
                                              shuffle=False)

    ####################
    input_dim = 4
    hidden_dim = 32
    num_layers = 4
    output_dim = 1
    num_epochs = 10
    # Here we define our model as a class

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print(model)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs.flatten(), labels.flatten())
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:  # print every 2000 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                #       (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
        model.eval()
        y_train_pred = model(X_test)
        pred = np.round(torch.sigmoid(y_train_pred.detach().squeeze()))
        print("acc:",accuracy_score(y_test.squeeze().flatten(), pred.flatten()))
        model.train()
        # print(running_loss)


    print('Finished Training')

    #%%


    y_train_pred = model(X_train)
    print(y_train_pred)

    pass

#%%
if __name__ == '__main__':
    main()
