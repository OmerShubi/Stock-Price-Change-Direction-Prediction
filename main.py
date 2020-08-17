# %%
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

from LSTM import LSTM_phase
from perceptron import perceptron_phase
from preprocessing import load_data
import matplotlib.pyplot as plt


def plots(df_day):
    df = df_day.copy()
    df = df.set_index(df_day['Date'])

    fig, axs = plt.subplots(1, 1)
    df['Open'].plot(ax=axs, title='IBM Stock Price, 1960 - 2020')
    axs.set_ylabel("Stock Price [USD]")

    plt.show()

    fig, axs = plt.subplots(1, 1)
    df_day.groupby(by='direction').count()['Date'].plot.bar(rot=0, ax=axs, title='Upward / Downward Days')
    axs.set_ylabel("Number of Days")
    axs.set_xticklabels(labels=['Down', 'Up'])

    plt.show()


def main():
    # load data
    df_day, week_features, week_targets = load_data()

    # Plot Number of Upward / Downward days, and stock price
    plots(df_day)

    # Perceptron
    # perceptron_phase(df_day)

    # LSTM
    # LSTM_phase(week_features, week_targets)


if __name__ == '__main__':
    main()
