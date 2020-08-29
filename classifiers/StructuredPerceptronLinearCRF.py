from pystruct.models import ChainCRF
from pystruct.learners import StructuredPerceptron
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging.config

from utils.Params import TEST_SIZE, SHUFFLE_TRAIN_TEST, STRUCT_PERCEPTRON_AVG, STRUCT_PERCEPTRON_MAXITER, FEATURES


def PerceptronCRF(week_features, week_targets):
    """
    Trains a perceptron based on a chain CRF
     on a training set and computes the accuracy on both the training and a validation set

    :param week_features: ndarray, 3d-array of week, day of week, day features
    :param week_targets: ndarray, 2d-array of week, direction of each day
    """
    logger = logging.getLogger(__name__)

    logger.info("------------- Structured Perceptron Phase -------------")
    logger.info(f"max iter:{STRUCT_PERCEPTRON_MAXITER},"
                f"avg from iter:{STRUCT_PERCEPTRON_AVG}")

    X_train, X_test, y_train, y_test = train_test_split(week_features,
                                                        week_targets,
                                                        test_size=TEST_SIZE,
                                                        random_state=42,
                                                        shuffle=SHUFFLE_TRAIN_TEST)

    model = ChainCRF(directed=True)
    clf = StructuredPerceptron(model=model, average=STRUCT_PERCEPTRON_AVG, max_iter=STRUCT_PERCEPTRON_MAXITER)
    clf.fit(X=X_train, Y=y_train.astype(int))
    logger.info(f"Structured Perceptron Train Accuracy:{round(clf.score(X_train, y_train.astype(int)), 3)}")
    logger.info(f"Structured Perceptron Test Accuracy:{round(clf.score(X_test, y_test.astype(int)), 3)}")
    # plt.plot(clf.loss_curve_)
    # plt.title('StructuredPerceptron Loss Curve')
    # plt.show()
