import pandas as pd
import numpy as np

from utils.Params import FEATURES, LOWER_THRESHOLD, UPPER_THRESHOLD
from sklearn.preprocessing import MinMaxScaler
from utils.Utils import plot_time_price, plot_time_volume, plot_direction_count
import logging.config


def load_data(file_path, minimum="", maximum="", use_preloaded=False):
    """
    Load the data if exists, otherwise create it using create_data function.

    :param file_path: path to stock data
    :param minimum: timestamp, keeps only dates after this day (for part 2)
    :param maximum: timestamp, keeps only dates before this day (for part 2)
    :param use_preloaded: bool, if False creates the data even if it exists.
    :return: day_features, day_target, week_features, week_targets, arr_change -
    ndarrays of the features and targets for both parts
    """
    logger = logging.getLogger(__name__)

    company = file_path.split("/")[2].split(".")[0]
    if use_preloaded:
        try:
            day_features = np.load(f'./data/{company}_day_features_{minimum}_{maximum}.npy')
            day_targets = np.load(f'./data/{company}_day_targets_{minimum}_{maximum}.npy')
            day_targets2 = np.load(f'./data/{company}_day_targets2_{minimum}_{maximum}.npy')
            week_features = np.load(f'./data/{company}_week_features_{minimum}_{maximum}.npy')
            week_targets = np.load(f'./data/{company}_week_targets_{minimum}_{maximum}.npy')
            week_targets2 = np.load(f'./data/{company}_week_targets2_{minimum}_{maximum}.npy')
            arr_change = np.load(f'./data/{company}_day_change_{minimum}_{maximum}.npy')

            logger.info(f'loading {company} data')
            return day_features, day_targets, week_features, week_targets, arr_change, day_targets2, week_targets2

        except FileNotFoundError as e:
            logger.info(e)
            logger.info(f'creating {company} data')
            return _create_data(file_path=file_path, company=company, minimum=minimum, maximum=maximum)
    else:
        logger.info(f'creating {company} data')
        return _create_data(file_path=file_path, company=company, minimum=minimum, maximum=maximum)


def _create_data(file_path, company, minimum=None, maximum=None):
    """
    Does the preprocessing by
        calling _preprocess_to_day,
        calling _preprocess_to_week,
        MinMax Scaling,
        filter by dates,
        plots graphs,
        saves processed data to disk

    :param file_path: path to load raw stock data from
    :param company: string, name of company
    :param minimum: timestamp, keeps only dates after this day (for part 2)
    :param maximum: timestamp, keeps only dates before this day (for part 2)
    :return:
    """
    logger = logging.getLogger(__name__)

    df = pd.read_csv(file_path, parse_dates=['Date'], index_col=['index'])
    df_day = _preprocess_to_day(df, minimum)
    df_day.to_csv('df_day_2.csv')
    # plot_time_price(df_day, title='IBM Stock Price, 1960 - 2020')
    # plot_time_volume(df_day, title='IBM Stock Volume, 1960 - 2020')

    scaler = MinMaxScaler()
    df_day[FEATURES + ['Change']] = scaler.fit_transform(df_day[FEATURES + ['Change']])

    week_features, week_targets, week_targets2 = _preprocess_to_week(df_day)
    logger.info(f'count of each direction2: {np.unique(week_targets2,return_counts=True)}')

    np.save(f'./data/{company}_week_features_{minimum}_{maximum}.npy', week_features)
    np.save(f'./data/{company}_week_targets_{minimum}_{maximum}.npy', week_targets)
    np.save(f'./data/{company}_week_targets2_{minimum}_{maximum}.npy', week_targets2)

    if minimum:
        df_day = df_day[df_day['Date'].between(minimum, maximum)]
        df_day = df_day[~(df_day['Date'] == '2011-02-17')]
        # df_day = df_day[~(df_day['Date'] == '1998-10-29')]
        logger.info(f'Num of days after drop {len(df_day)}')
        df_day = df_day.sort_values(by='Date')

    df_change = df_day['Change'].to_numpy()
    np.save(f'./data/{company}_day_change_{minimum}_{maximum}.npy', df_change)

    day_features = df_day[FEATURES].to_numpy()
    day_targets = df_day['direction'].to_numpy()
    day_targets2 = df_day['direction2'].to_numpy()
    np.save(f'./data/{company}_day_features_{minimum}_{maximum}.npy', day_features)
    np.save(f'./data/{company}_day_targets_{minimum}_{maximum}.npy', day_targets)
    np.save(f'./data/{company}_day_targets2_{minimum}_{maximum}.npy', day_targets2)

    return day_features, day_targets, week_features, week_targets, df_change, day_targets2, week_targets2


def _preprocess_to_day(input_df, compute_statistics):
    """
    Given raw input dataframe -
        Compute Direction
        Compute Change
        Drop OpenInt & Close Columns

    :param input_df: raw input dataframe
    :param compute_statistics: bool, whether to compute date range and number of days or not
    :return: copy of input_df after transformation
    """
    logger = logging.getLogger(__name__)

    df = input_df.copy()
    # check OpenInt values
    # if df[df.OpenInt > 0].empty: logger.info("all OpenInt values are zero")
    # remove OpenInt column
    df.drop("OpenInt", axis=1, inplace=True)
    # remove High, low columns
    # df.drop(["High",'Low'],axis=1, inplace=True)
    # add direction column : 1 if Close price >= Open price else 0
    df.loc[df.Close >= df.Open, 'direction'] = 1
    df.loc[df.Close < df.Open, 'direction'] = 0

    df['Change'] = df.Close - df.Open

    df['percent_change'] = df['Change'] / df['Open']

    df.loc[df.percent_change < LOWER_THRESHOLD, 'direction2'] = 0
    df.loc[df.percent_change.between(LOWER_THRESHOLD, 0), 'direction2'] = 1
    df.loc[df.percent_change.between(0, UPPER_THRESHOLD), 'direction2'] = 2
    df.loc[df.percent_change > UPPER_THRESHOLD, 'direction2'] = 3
    # x < -.05 : 0, -.05<=x<0 : 1, 0<=x<=.5 : 2 , x> .5 : 3

    df.drop(["Close"], axis=1, inplace=True)
    df['Low_mid'] = df['Low'] + 0.5 * (df['Open'] - df['Low'])
    df['High_mid'] = df['High'] - 0.5 * (df['High'] - df['Open'])
    # check days range data
    if not compute_statistics:
        logger.info(f'Date range: {min(df.Date)}-{max(df.Date)}')
        logger.info(f'Num of days:{len(df)}')
    return df


def _preprocess_to_week(input_df):
    """
    Given a day_df dataframe, create weekly features and targets.
        Weekly features are the features of each day in a week.
        Only keep 'full weeks' (5 days, monday-thursday)
        targets are the direction of each day

    :param input_df: a dataframe that has date, FEATURES and direction columns
    :return: features, targets - ndarrays.
                features is a 3d-array of week, day of week, day features
                targets is a 2d-array of week, direction of each day
    """
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
    df = df[FEATURES + ['weekNum', 'direction', 'direction2']]
    num_of_weeks = max(df.weekNum)
    features = []
    targets = []
    targets2 = []
    features_len = len(FEATURES)
    for week_num in range(int(num_of_weeks)):
        week_feature_list = []
        week_label_list = []
        week_label_list2 = []
        week = df[df['weekNum'] == week_num]
        for _, day in week.iterrows():
            week_feature_list.append(day[0:features_len].to_numpy())
            week_label_list.append(day[-2])
            week_label_list2.append(day[-1])
        features.append(week_feature_list)
        targets.append(week_label_list)
        targets2.append(week_label_list2)
    return np.array(features), np.array(targets), np.array(targets2)
