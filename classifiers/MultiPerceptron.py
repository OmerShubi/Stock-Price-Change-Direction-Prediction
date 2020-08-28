from sklearn.neural_network import MLPClassifier
from utils.Params import MLP_MAXITER
import logging.config


def multi_perceptron_phase(X_train, X_test, y_train, y_test, pair):
    """
    TODO
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param pair:
    :return:
    """
    logger = logging.getLogger(__name__)

    model = MLPClassifier(random_state=0, n_iter_no_change=100, learning_rate='adaptive', max_iter=MLP_MAXITER)
    model.fit(X_train, y_train)
    predictions_train = model.predict_proba(X_train)
    predictions_test = model.predict_proba(X_test)
    logger.info(f"MultiPerceptron accuracy for {pair}:"
                f" Train:{round(model.score(X_train, y_train),3)},"
                f" Test: {round(model.score(X_test, y_test),3)}")
    return predictions_train, predictions_test

