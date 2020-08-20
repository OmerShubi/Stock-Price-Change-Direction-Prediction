from sklearn.neural_network import MLPClassifier
from utils.Params import FEATURES
from utils.Params import TEST_SIZE, SHUFFLE_TRAIN_TEST, MLP_MAXITER

def multi_perceptron_phase(X_train, X_test, y_train, y_test, pair):
    print(f"--------- MultiPerceptron Phase for {pair}---------")
    print(f"Features:{FEATURES},"
          f"Test size:{TEST_SIZE},"
          f" Shuffle? {SHUFFLE_TRAIN_TEST}")

    model = MLPClassifier(random_state=0, n_iter_no_change=100, learning_rate ='adaptive', max_iter=MLP_MAXITER)
    model.fit(X_train, y_train)
    predictions_train = model.predict_proba(X_train)
    predictions_test = model.predict_proba(X_test)
    print(f"MultiPerceptron for {pair} Train accuracy {model.score(X_train, y_train)}")
    print(f"MultiPerceptron for {pair} Test accuracy {model.score(X_test, y_test)}")
    return predictions_train, predictions_test

