from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

from utils.params import TEST_SIZE, SHUFFLE_TRAIN_TEST
from utils.params import FEATURES


def perceptron_phase(X, y):
    print("--------- Perceptron Phase ---------")
    print(f"Features:{FEATURES},"
          f"Test size:{TEST_SIZE},"
          f" Shuffle? {SHUFFLE_TRAIN_TEST}")

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TEST_SIZE,
                                                        random_state=42,
                                                        shuffle=SHUFFLE_TRAIN_TEST)

    clf = Perceptron(random_state=0, n_iter_no_change=100)
    clf.fit(X_train, y_train)
    print("Perceptron Train Accuracy:", round(clf.score(X_train, y_train),3))
    print("Perceptron Test Accuracy:", round(clf.score(X_test, y_test), 3))

