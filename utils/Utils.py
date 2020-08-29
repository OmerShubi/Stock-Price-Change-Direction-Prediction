import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import logging.config


def plot_time_price(df_day, title):
    """
    Plots the Price vs time

    :param title: Title of the graph
    :param df_day: dataframe that has 'Open' Column
    :return: shows and saves the figure with title name
    """
    df = df_day.copy()
    df = df.set_index(df_day['Date'])

    fig, axs = plt.subplots(1, 1)
    df['Open'].plot(ax=axs, title=title, linewidth=0.5)
    df['High'].plot(ax=axs, linewidth=0.5)
    df['Low'].plot(ax=axs, linewidth=0.5)
    axs.set_ylabel("Stock Price [USD]")
    plt.savefig(f'{title}.png')
    plt.show()


def plot_time_volume(df_day, title):
    """
    Plots the Volume vs time

    :param title: Title of the graph
    :param df_day: dataframe that has 'Volume' Column
    :return: shows and saves the figure with title name
    """
    df = df_day.copy()
    df = df.set_index(df_day['Date'])

    fig, axs = plt.subplots(1, 1)
    df['Volume'].plot(ax=axs, title=title)
    axs.set_ylabel("Stock Volume [USD]")
    plt.savefig(f'{title}.png')
    plt.show()


def plot_direction_count(df_day):
    """
    Plots bar chart of how many up days and how many down days

    :param df_day: dataframe that has 'direction' Column
    """
    logger = logging.getLogger(__name__)

    logger.info(df_day.groupby(by='direction').count()['Date'])
    fig, axs = plt.subplots(1, 1)
    df_day.groupby(by='direction').count()['Date'].plot.bar(rot=0, ax=axs, title='Upward / Downward Days')
    axs.set_ylabel("Number of Days")
    axs.set_xticklabels(labels=['Down', 'Up'])

    plt.show()


def calc_correlation(arr1_change, arr2_change):
    """
    Calculcates the Pearson Correlation between to vectors
    :param arr1_change: ndarray consisting of one 1D array
    :param arr2_change: ndarray consisting of the other 1D array
    :return: float, Pearson Correlation
    """
    pearson = pearsonr(arr1_change, arr2_change)[0]
    return pearson


def find_dates(stock_names):
    """
    Finds the minimum and maximum dates that are available in all the given stocks

    :param stock_names: list of stock names to use
    :return: minimum, maximum, both timestamps
    """
    stocks_df = []
    for stock in stock_names:
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

    return minimum, maximum

# stock1 =stocks[0][0]
# stock2 =stocks[2][0]
# common = stock1.merge(stock2, on=["Date"])
# result = stock1[~stock1.Date.isin(common.Date)]
