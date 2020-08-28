from classifiers.LSTM import LSTM_phase
from classifiers.StructuredPerceptronLinearCRF import PerceptronCRF
from classifiers.Perceptron import perceptron_phase
from classifiers.MultiPerceptron import multi_perceptron_phase
from utils.Params import USE_PRELOADED, PERCEPTRON_TRAIN, STRUCT_PERCEPTRON_TRAIN, \
    LSTM_TRAIN, PART1, PART2, THRESHOLD, FEATURES, MLP_MAXITER
from utils.PreProcessing import load_data
from utils.Utils import calc_correlation, find_dates
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from utils.Params import TEST_SIZE, SHUFFLE_TRAIN_TEST
from classifiers.BeliefPropMRF import batch_infer
import logging.config


def main():
    """
    TODO
    :return:
    """
    # Gets or creates a logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger(__name__)

    if PART1:
        logger.info('--------- Part 1 Start ---------')
        logger.info(f"Features:{FEATURES},"
                    f"Test size:{TEST_SIZE},"
                    f" Shuffle? {SHUFFLE_TRAIN_TEST}")
        # PART 1

        # create/load data
        day_features, day_targets, week_features, week_targets, array_change = load_data(file_path='./data/ibm.us.txt',
                                                                                         use_preloaded=USE_PRELOADED)

        # Plot Number of Upward / Downward days,
        # plot_direction_count(df_day)

        # Perceptron
        if PERCEPTRON_TRAIN:
            perceptron_phase(day_features, day_targets)

        # PerceptronCRF
        if STRUCT_PERCEPTRON_TRAIN:
            PerceptronCRF(week_features, week_targets)

        # LSTM
        if LSTM_TRAIN:
            LSTM_phase(week_features, week_targets)

        logger.info('--------- Part 1 Finish ---------')

    if PART2:
        """
            correlation train between all two stock_names
            train perceptron for all two corrleated stock_names:
            example : [features-ibm, features-crm, label] , label - [(0,0),(0,1),(1,0),(1,1)]
            inference by perceptron o test -> potentail matrix
            graph for test
            BP
        """
        logger.info('--------- Part 2 Start ---------')
        stock_names = ['ibm', 'orcl', 'sap', 'csco', 'intc']
        logger.info(f'stocks: {stock_names}, Threshold:{THRESHOLD}')
        removed_stocks = ['acn', 'crm']
        minimum, maximum = find_dates(stock_names)
        logger.info(f'Date range: {minimum}-{maximum}')

        # create/load data
        stocks = []
        for stock_name in stock_names:
            array_features, array_targets, _, _, array_change = load_data(file_path=f'./data/{stock_name}.us.txt',
                                                                          use_preloaded=USE_PRELOADED,
                                                                          minimum=minimum,
                                                                          maximum=maximum)
            stocks.append((array_features, array_targets, array_change))

        logger.info(f'Number of days: {array_features.shape[0]}')
        PairsDataTrain = {}
        PairsDataTest = {}
        StockDataTrain = {}
        StockDataTest = {}
        PairsPearson = {}
        logger.info(f"--------- MultiPerceptron Phase  ---------")
        logger.info(f"Max Iter:{MLP_MAXITER}")

        # Compute Pairwise Correlations, train MLP for pairs of companies above threshold
        for pair in combinations(zip(stock_names, stocks), 2):
            stock1, stock2 = pair
            stock1_name, stock1_data = stock1
            stock2_name, stock2_data = stock2
            pair_name = (stock1_name, stock2_name)

            stock1_features, stock1_targets, stock1_change = stock1_data
            stock2_features, stock2_targets, stock2_change = stock2_data

            pearson = calc_correlation(stock1_change, stock2_change)
            PairsPearson[pair_name] = round(pearson, 3)

            if pearson >= THRESHOLD:
                stock1_features_train, stock1_features_test, stock1_targets_train, stock1_targets_test = \
                    train_test_split(stock1_features,
                                     stock1_targets,
                                     test_size=TEST_SIZE,
                                     random_state=42,
                                     shuffle=SHUFFLE_TRAIN_TEST)

                stock2_features_train, stock2_features_test, stock2_targets_train, stock2_targets_test = \
                    train_test_split(stock2_features,
                                     stock2_targets,
                                     test_size=TEST_SIZE,
                                     random_state=42,
                                     shuffle=SHUFFLE_TRAIN_TEST)

                # StockData[stock] = X_train, y_train, X_test, y_test
                StockDataTrain[stock1_name] = (stock1_features_train, stock1_targets_train)
                StockDataTest[stock1_name] = (stock1_features_test, stock1_targets_test)
                StockDataTrain[stock2_name] = (stock2_features_train, stock2_targets_train)
                StockDataTest[stock2_name] = (stock2_features_test, stock2_targets_test)

                X_train = np.concatenate((stock1_features_train, stock2_features_train), axis=1)
                num_days_train = X_train.shape[0]
                X_test = np.concatenate((stock1_features_test, stock2_features_test), axis=1)
                num_days_test = X_test.shape[0]

                # 0:(0,0) 1:(0,1) 2: (1,0) 3 : (1,1), 1 means stock price direction is up, 0 down
                y_train = 2 * stock1_targets_train + stock2_targets_train
                y_test = 2 * stock1_targets_test + stock2_targets_test

                # PairsData[pair] = predictions_train, predictions_test
                PairsDataTrain[pair_name], PairsDataTest[pair_name] = multi_perceptron_phase(X_train,
                                                                                             X_test,
                                                                                             y_train,
                                                                                             y_test,
                                                                                             pair_name)

        logger.info(f"Pearson Correlations:{PairsPearson}")
        batch_infer(PairsDataTrain, StockDataTrain, num_days_train, 'train')
        batch_infer(PairsDataTest, StockDataTest, num_days_test, 'test')

        logger.info('--------- Part 2 Finished ---------')
    if not PART1 and not PART2:
        logger.info('No parts :(')


if __name__ == '__main__':
    main()
