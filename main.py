from classifiers.LSTM import LSTM_phase
from classifiers.StructuredPerceptronLinearCRF import PerceptronCRF
from classifiers.perceptron import perceptron_phase
from utils.params import USE_PRELOADED, PERCEPTRON_TRAIN, STRUCT_PERCEPTRON_TRAIN, LSTM_TRAIN
from utils.preprocessing import load_data
import matplotlib.pyplot as plt


# def plots(df_day):
#
#
#     print(df_day.groupby(by='direction').count()['Date'])
#     fig, axs = plt.subplots(1, 1)
#     df_day.groupby(by='direction').count()['Date'].plot.bar(rot=0, ax=axs, title='Upward / Downward Days')
#     axs.set_ylabel("Number of Days")
#     axs.set_xticklabels(labels=['Down', 'Up'])
#
#     plt.show()


def main():
    # load data
    day_features, day_targets, week_features, week_targets = load_data(use_preloaded=USE_PRELOADED)

    # Plot Number of Upward / Downward days,
    # plots(df_day)

    # Perceptron
    if PERCEPTRON_TRAIN:
        perceptron_phase(day_features, day_targets)

    # PerceptronCRF
    if STRUCT_PERCEPTRON_TRAIN:
        PerceptronCRF(week_features, week_targets)

    # LSTM
    if LSTM_TRAIN:
        LSTM_phase(week_features, week_targets)


if __name__ == '__main__':
    main()
