import logging.config

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

from utils.Params import TEST_SIZE, SHUFFLE_TRAIN_TEST, PART3
from utils.Utils import compute_prediction_report


def perceptron_phase(X, y, y2=None):
    """
    Trains a perceptron on a training set and computes the accuracy on both a training and a validation set

    :param X: ndarray, each row represents a day, each column a feature
    :param y: ndarray, each row is the target of the corresponding day, 1 if direction is up, 0 otherwise
    """
    logger = logging.getLogger(__name__)

    logger.info("--------- Perceptron Phase ---------")
    is_part1 = y2 is not None
    if y2 is None:
        y2 = y.copy()

    X_train, X_test, y_train, y_test, y2_train, y2_test = train_test_split(X, y, y2,
                                                                           test_size=TEST_SIZE,
                                                                           random_state=42,
                                                                           shuffle=SHUFFLE_TRAIN_TEST)

    clf = Perceptron(random_state=0, n_iter_no_change=100)
    clf.fit(X_train, y_train)
    logger.info(f"Perceptron Train Accuracy: {round(clf.score(X_train, y_train), 3)}")
    logger.info(f"Perceptron Test Accuracy: {round(clf.score(X_test, y_test), 3)}")
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    compute_prediction_report(y_train_pred, y2_train, y_train, is_part1)
    compute_prediction_report(y_test_pred, y2_test, y_test, is_part1)
