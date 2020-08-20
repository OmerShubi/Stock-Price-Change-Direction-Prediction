import pandas as pd
import numpy as np

from utils.Params import FEATURES
from sklearn.preprocessing import MinMaxScaler
from utils.Utils import plot_time_price, plot_time_volume, plot_direction_count


def preprocess_to_day(input_df):
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
    df['Change'] = df.Close - df.Open
    df.drop(["Close"], axis=1, inplace=True)
    # check days range data
    print(f'Data from day {min(df.Date)} to day {max(df.Date)}')
    print('Num of days before drop', len(df))

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
    return np.array(features), np.array(targets)

def load_data(file_path, minimum=None ,maximum=None , use_preloaded=False):
    company = file_path.split("/")[2].split(".")[0]
    if use_preloaded:
        try:
            day_features = np.load(f'./data/{company}_day_features.npy')
            day_targets = np.load(f'./data/{company}_day_targets.npy')
            week_features = np.load(f'./data/{company}_week_features.npy')
            week_targets = np.load(f'./data/{company}_week_targets.npy')
            df_change = np.load(f'./data/{company}_day_change.npy')

            print(f'loading {company} data')
            return day_features, day_targets, week_features, week_targets, df_change

        except FileNotFoundError as e:
            print(e)
            print(f'creating {company} data')
            return create_data(file_path=file_path, company=company, minimum=minimum, maximum=maximum)
    else:
        print(f'creating {company} data')
        return create_data(file_path=file_path, company=company, minimum=minimum, maximum=maximum)

def create_data(file_path, company, minimum=None, maximum=None):
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col=['index'])
    df_day = preprocess_to_day(df)

    # plot_time_price(df_day)
    # plot_time_volume(df_day)

    scaler = MinMaxScaler()
    df_day[FEATURES] = scaler.fit_transform(df_day[FEATURES])

    week_features, week_targets = preprocess_to_week(df_day)
    np.save(f'./data/{company}_week_features.npy', week_features)
    np.save(f'./data/{company}_week_targets.npy', week_targets)

    if minimum:
        df_day = df_day[df_day['Date'].between(minimum, maximum)]
        df_day = df_day[~(df_day['Date'] == '2011-02-17')]
        # df_day = df_day[~(df_day['Date'] == '1998-10-29')]
        print('Num of days after drop', len(df_day))
        df_day = df_day.sort_values(by='Date')

    df_change = df_day['Change'].to_numpy()
    np.save(f'./data/{company}_day_change.npy', df_change)

    day_features = df_day[FEATURES].to_numpy()
    day_target = df_day['direction'].to_numpy()
    np.save(f'./data/{company}_day_features.npy', day_features)
    np.save(f'./data/{company}_day_targets.npy', day_target)

    return day_features, day_target, week_features, week_targets, df_change

