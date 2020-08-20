import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import logging.config


def plot_time_price(df_day):
    df = df_day.copy()
    df = df.set_index(df_day['Date'])

    fig, axs = plt.subplots(1, 1)
    df['Open'].plot(ax=axs, title='IBM Stock Price, 1960 - 2020', linewidth=0.5)
    df['High'].plot(ax=axs, linewidth=0.5)
    df['Low'].plot(ax=axs, linewidth=0.5)
    axs.set_ylabel("Stock Price [USD]")
    plt.savefig('IBM_Stock_Price_1960_2020.png')
    plt.show()


def plot_time_volume(df_day):
    df = df_day.copy()
    df = df.set_index(df_day['Date'])

    fig, axs = plt.subplots(1, 1)
    df['Volume'].plot(ax=axs, title='IBM Stock Volume, 1960 - 2020')

    axs.set_ylabel("Stock Volume [USD]")
    plt.savefig('IBM_Stock_Volume_1960_2020.png')
    plt.show()


def plot_direction_count(df_day):
    logger = logging.getLogger(__name__)

    logger.info(df_day.groupby(by='direction').count()['Date'])
    fig, axs = plt.subplots(1, 1)
    df_day.groupby(by='direction').count()['Date'].plot.bar(rot=0, ax=axs, title='Upward / Downward Days')
    axs.set_ylabel("Number of Days")
    axs.set_xticklabels(labels=['Down', 'Up'])

    plt.show()


def calc_correlation(df1_change, df2_change):

    pearson = pearsonr(df1_change, df2_change)[0]
    return pearson


def find_dates(stocks):
    stocks_df = []
    for stock in stocks:
        df = pd.read_csv(f'./data/{stock}.us.txt', parse_dates=['Date'], index_col=['index'])
        stocks_df.append(df)
    for index, df in enumerate(stocks_df):
        if index == 0:
            minimum = min(df.Date)
            maximum = max(df.Date)
        if index != 0 and min(df.Date) > minimum:
            minimum = min(df.Date)
        if index != 0 and max(df.Date) < maximum:
            maximum = max(df.Date)
    # stocks_df_slice = []
    # for df in stocks_df:
    #     stocks_df_slice.append(df[df['Date'].between(minimum, maximum)])
    return minimum, maximum
