from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from utils.Params import TEST_SIZE, SHUFFLE_TRAIN_TEST
from utils.Params import FEATURES

def multi_perceptron_phase(X, y, pair):
    print("--------- Perceptron Phase ---------")
    print(f"Features:{FEATURES},"
          f"Test size:{TEST_SIZE},"
          f" Shuffle? {SHUFFLE_TRAIN_TEST}")

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TEST_SIZE,
                                                        random_state=42,
                                                        shuffle=SHUFFLE_TRAIN_TEST)

    model = MLPClassifier(random_state=0, n_iter_no_change=100)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)
    print(f"MultiPerceptron for {pair} Train accuracy {model.score(X_train, y_train)}")
    print(f"MultiPerceptron for {pair} Test accuracy {model.score(X_test, y_test)}")
    return X_test, y_test, predictions

