
########## ALL ##########
TEST_SIZE = 0.2
SHUFFLE_TRAIN_TEST = True
batch_size = 16

## IF CHANGE FEATURES USE PRELOADE FALSE ONE TIME
FEATURES = ['Open', 'High', 'Low', 'Volume']
USE_PRELOADED = False

########## Perceptron #########
PERCEPTRON_TRAIN = True

########## Perceptron CRF ##########
STRUCT_PERCEPTRON_TRAIN = True
STRUCT_PERCEPTRON_AVG = 10
STRUCT_PERCEPTRON_MAXITER = 200

########## LSTM ##########
LSTM_TRAIN = True
input_dim = len(FEATURES)
hidden_dim = 120
num_layers = 2
output_dim = 1
num_epochs = 200
