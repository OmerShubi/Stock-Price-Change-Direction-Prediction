from classifiers.LSTM import LSTM_phase
from classifiers.StructuredPerceptronLinearCRF import PerceptronCRF
from classifiers.Perceptron import perceptron_phase
from classifiers.MultiPerceptron import multi_perceptron_phase
from utils.Params import USE_PRELOADED, PERCEPTRON_TRAIN, STRUCT_PERCEPTRON_TRAIN,\
    LSTM_TRAIN, PART1, PART2, THRESHOLD
from utils.PreProcessing import load_data
from utils.Utils import  calc_correlaction, find_dates
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from utils.Params import TEST_SIZE, SHUFFLE_TRAIN_TEST
from classifiers.BeliefPropMRF import batch_infer



def main():
    if PART1:
        # PART 1

        # create/load data
        day_features, day_targets, week_features, week_targets = load_data(file_path='./data/ibm.us.txt',
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

    elif PART2:
        # correlation train between all two stocks
        # train perceptron for all two corrleated stocks:
            # example : [features-ibm, features-crm, label] , label - [(0,0),(0,1),(1,0),(1,1)]
        # inference by perceptron o test -> potentail matrix
        # graph for test
        # BP


        # PART 2
        stocks = ['ibm', 'crm', 'sap']#, 'orcl', 'msft', 'acn']
        minimum, maximum = find_dates(stocks)

        # create/load data
        df_features_ibm, df_targets_ibm, _, _ = load_data(file_path='./data/ibm.us.txt',
                                                          use_preloaded=USE_PRELOADED,
                                                          minimum=minimum,
                                                          maximum=maximum)
        df_features_crm, df_targets_crm, _, _ = load_data(file_path='./data/crm.us.txt',
                                                          use_preloaded=USE_PRELOADED,
                                                          minimum=minimum,
                                                          maximum=maximum)
        df_features_sap, df_targets_sap, _, _ = load_data(file_path='./data/sap.us.txt',
                                                          use_preloaded=USE_PRELOADED,
                                                          minimum=minimum,
                                                          maximum=maximum)
        df_features_orcl, df_targets_orcl, _, _ = load_data(file_path='./data/orcl.us.txt',
                                                            use_preloaded=USE_PRELOADED,
                                                            minimum=minimum,
                                                            maximum=maximum)
        df_features_msft, df_targets_msft, _, _ = load_data(file_path='./data/msft.us.txt',
                                                            use_preloaded=USE_PRELOADED,
                                                            minimum=minimum,
                                                            maximum=maximum)
        df_features_acn, df_targets_acn, _, _ = load_data(file_path='./data/acn.us.txt',
                                                          use_preloaded=USE_PRELOADED,
                                                          minimum=minimum,
                                                          maximum=maximum)
        PairsDataTrain = {}
        PairsDataTest = {}
        StockDataTrain = {}
        StockDataTest = {}
        for pair in combinations(stocks,2):
            if calc_correlaction(pair) >= THRESHOLD:
                if USE_PRELOADED:
                    df1_features = eval(f"df_features_{pair[0]}")
                    df1_targets = eval(f"df_targets_{pair[0]}")
                    df2_features = eval(f"df_features_{pair[1]}")
                    df2_targets = eval(f"df_targets_{pair[1]}")
                else:
                    df1_features = eval(f"df_features_{pair[0]}").to_numpy()
                    df1_targets = eval(f"df_targets_{pair[0]}").to_numpy()
                    df2_features = eval(f"df_features_{pair[1]}").to_numpy()
                    df2_targets = eval(f"df_targets_{pair[1]}").to_numpy()

                df1_features_train, df1_features_test, df1_targets_train,\
                df1_targets_test = train_test_split(df1_features,
                                                    df1_targets,
                                                    test_size=TEST_SIZE,
                                                    random_state=42,
                                                    shuffle=SHUFFLE_TRAIN_TEST)

                df2_features_train, df2_features_test, df2_targets_train, \
                df2_targets_test = train_test_split(df2_features,
                                                    df2_targets,
                                                    test_size=TEST_SIZE,
                                                    random_state=42,
                                                    shuffle=SHUFFLE_TRAIN_TEST)

                # StockData[stock] = X_train, y_train, X_test, y_test
                StockDataTrain[pair[0]] = (df1_features_train, df1_targets_train)
                StockDataTest[pair[0]] = (df1_features_test, df1_targets_test)
                StockDataTrain[pair[1]] = (df2_features_train, df2_targets_train)
                StockDataTest[pair[1]] = (df2_features_test, df2_targets_test)


                X_train = np.concatenate((df1_features_train, df2_features_train), axis=1)
                num_days_train = X_train.shape[0]
                X_test = np.concatenate((df1_features_test, df2_features_test), axis=1)
                num_days_test = X_test.shape[0]

                # 0:(0,0) 1:(0,1) 2: (1,0) 3 : (1,1)
                y_train = 2 * df1_targets_train + df2_targets_train
                y_test = 2 * df1_targets_test + df2_targets_test

                # PairsData[pair] = predictions_train, predictions_test
                PairsDataTrain[pair], PairsDataTest[pair] = multi_perceptron_phase(X_train, X_test, y_train, y_test, pair)

        batch_infer(PairsDataTrain, StockDataTrain, num_days_train, 'train')
        batch_infer(PairsDataTest, StockDataTest, num_days_test, 'test')


    else:
        print("No Part to Run ..")


if __name__ == '__main__':
    main()
