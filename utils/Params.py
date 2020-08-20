########## ALL ##########
TEST_SIZE = 0.2
PART1 = False
PART2 = True

## IF CHANGE FEATURES USE PRELOADE FALSE ONE TIME
FEATURES = ['Open', 'High', 'Low', 'Volume']
SHUFFLE_TRAIN_TEST = True
USE_PRELOADED = True

########## Perceptron #########
PERCEPTRON_TRAIN = True

########## Perceptron CRF ##########
STRUCT_PERCEPTRON_TRAIN = True
STRUCT_PERCEPTRON_AVG = 10
STRUCT_PERCEPTRON_MAXITER = 200

########## LSTM ##########
LSTM_TRAIN = True
hidden_dim = 120
num_layers = 2
num_epochs = 2
batch_size = 16
input_dim = len(FEATURES)  # do not change
output_dim = 1  # do not change

######## MLP #######
MLP_MAXITER = 2

###### MRF #####
THRESHOLD = 0.5
