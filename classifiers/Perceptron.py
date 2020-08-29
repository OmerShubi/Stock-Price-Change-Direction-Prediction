import logging.config

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

from utils.Params import TEST_SIZE, SHUFFLE_TRAIN_TEST


def perceptron_phase(X, y):
    """
    Trains a perceptron on a training set and computes the accuracy on both a training and a validation set

    :param X: ndarray, each row represents a day, each column a feature
    :param y: ndarray, each row is the target of the corresponding day, 1 if direction is up, 0 otherwise
    """
    logger = logging.getLogger(__name__)

    logger.info("--------- Perceptron Phase ---------")

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TEST_SIZE,
                                                        random_state=42,
                                                        shuffle=SHUFFLE_TRAIN_TEST)

    clf = Perceptron(random_state=0, n_iter_no_change=100)
    clf.fit(X_train, y_train)
    logger.info(f"Perceptron Train Accuracy: {round(clf.score(X_train, y_train), 3)}")
    logger.info(f"Perceptron Test Accuracy: {round(clf.score(X_test, y_test), 3)}")
