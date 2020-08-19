from pystruct.models import ChainCRF
from pystruct.learners import StructuredPerceptron
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils.Params import TEST_SIZE, SHUFFLE_TRAIN_TEST, STRUCT_PERCEPTRON_AVG, STRUCT_PERCEPTRON_MAXITER


def PerceptronCRF(week_features, week_targets):
    print("------------- Structured Perceptron Phase -------------")
    X_train, X_test, y_train, y_test = train_test_split(week_features,
                                                        week_targets,
                                                        test_size=TEST_SIZE,
                                                        random_state=42,
                                                        shuffle=SHUFFLE_TRAIN_TEST)

    model = ChainCRF(directed=True)
    clf = StructuredPerceptron(model=model, average=STRUCT_PERCEPTRON_AVG, max_iter=STRUCT_PERCEPTRON_MAXITER)  # ,decay_exponent=0.9)
    clf.fit(X=X_train, Y=y_train.astype(int))
    print("Structured Perceptron Train Accuracy:", round(clf.score(X_train, y_train.astype(int)), 3))
    print("Structured Perceptron Test Accuracy:", round(clf.score(X_test, y_test.astype(int)), 3))
    plt.plot(clf.loss_curve_)
    plt.show()
    pass