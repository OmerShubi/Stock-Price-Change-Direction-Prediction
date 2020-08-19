from classifiers.LSTM import LSTM_phase
from classifiers.StructuredPerceptronLinearCRF import PerceptronCRF
from classifiers.perceptron import perceptron_phase
from utils.params import USE_PRELOADED, PERCEPTRON_TRAIN, STRUCT_PERCEPTRON_TRAIN, LSTM_TRAIN, PART1, PART2
from utils.preprocessing import load_data, find_dates


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
        files = ['./data/ibm.us.txt', './data/crm.us.txt', './data/sap.us.txt',
                 './data/orcl.us.txt', './data/msft.us.txt', './data/acn.us.txt']
        minimum, maximum = find_dates(files)

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

        pass


    else:
        print("No Part to Run ..")


if __name__ == '__main__':
    main()
