from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

from params import TEST_SIZE, SHUFFLE_TRAIN_TEST
from params import FEATURES


def perceptron_phase(df_day):
    print("--------- Perceptron Phase ---------")
    print(f"Features:{FEATURES},"
          f"Test size:{TEST_SIZE},"
          f" Shuffle? {SHUFFLE_TRAIN_TEST}")
    X = df_day[FEATURES]
    y = df_day['direction']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TEST_SIZE,
                                                        random_state=42,
                                                        shuffle=SHUFFLE_TRAIN_TEST)

    clf = Perceptron(random_state=0)
    clf.fit(X_train, y_train)
    print("Perceptron Accuracy:", clf.score(X_test, y_test))

