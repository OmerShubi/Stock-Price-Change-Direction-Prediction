from sklearn.neural_network import MLPClassifier
from utils.Params import MLP_MAXITER
import logging.config


def multi_perceptron_phase(X_train, X_test, y_train, y_test, pair):
    """
    Trains a Multi-Layer perceptron
    on a training set and computes the accuracy on both the training and a validation set

    Features are a concatenation of features of both companies
    Categorical Labels are:
        0 if both companies have direction down
        1 if company 1 has up direction and company 2 has down direction
        2 if company 1 has down direction and company 2 has up direction
        3 if both companies have direction up

    :param X_train: ndarray, training set, each row represents a day, each column a feature
    :param X_test: ndarray, test set, each row is the target of the corresponding day
    :param y_train: ndarray, training set label, each row represents a day, each column a feature
    :param y_test: ndarray, training set label,each row is the target of the corresponding day
    :param pair: string, names of the two companies
    :return: predicted probability for each of the four options (labels),
    for each day, both in train set (ndarray, predictions_train) and in test set (ndarray, predictions_test)
    """
    logger = logging.getLogger(__name__)

    model = MLPClassifier(random_state=0, n_iter_no_change=100, learning_rate='adaptive', max_iter=MLP_MAXITER)
    model.fit(X_train, y_train)
    predictions_train = model.predict_proba(X_train)
    predictions_test = model.predict_proba(X_test)
    logger.info(f"MultiPerceptron accuracy for {pair}:"
                f" Train:{round(model.score(X_train, y_train), 3)},"
                f" Test: {round(model.score(X_test, y_test), 3)}")
    return predictions_train, predictions_test
