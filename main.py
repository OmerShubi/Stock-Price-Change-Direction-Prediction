from classifiers.LSTM import LSTM_phase
from classifiers.StructuredPerceptronLinearCRF import PerceptronCRF
from classifiers.Perceptron import perceptron_phase
from classifiers.MultiPerceptron import multi_perceptron_phase
from utils.Params import USE_PRELOADED, PERCEPTRON_TRAIN, STRUCT_PERCEPTRON_TRAIN,\
    LSTM_TRAIN, PART1, PART2, THRESHOLD
from utils.PreProcessing import load_data, find_dates, calc_correlaction
from itertools import combinations


def main():
    if PART1:
        # PART 1

        # create/load data
        day_features, day_targets, week_features, week_targets = load_data(file_path='./data/ibm.us.txt',
                                                                           use_preloaded=USE_PRELOADED)

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

    elif PART2:
        # correlation train between all two stocks
        # train perceptron for all two corrleated stocks:
            # example : [features-ibm, features-crm, label] , label - [(0,0),(0,1),(1,0),(1,1)]
        # inference by perceptron o test -> potentail matrix
        # graph for test
        # BP


        # PART 2
        stocks = ['ibm', 'crm', 'sap', 'orcl', 'msft', 'acn']
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
        import numpy as np
        PairsDate = {}
        for pair in combinations(stocks,2):
            if calc_correlaction(pair) >= THRESHOLD:
                df1_features = eval(f"df_features_{pair[0]}").to_numpy()
                df1_targets = eval(f"df_targets_{pair[0]}").to_numpy()
                df2_features = eval(f"df_features_{pair[1]}").to_numpy()
                df2_targets = eval(f"df_targets_{pair[1]}").to_numpy()
                df_features = np.concatenate((df1_features, df2_features), axis=1)
                # 0:(0,0) 1:(0,1) 2: (1,0) 3 : (1,1)
                df_targets = 2*df1_targets + df2_targets
                # (X_test, y_test, predictions)
                PairsDate[pair] = multi_perceptron_phase(df_features, df_targets, pair)

    else:
        print("No Part to Run ..")


if __name__ == '__main__':
    main()
