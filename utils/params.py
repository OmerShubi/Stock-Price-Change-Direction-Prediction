TEST_SIZE = 0.2
SHUFFLE_TRAIN_TEST = False
batch_size = 16
FEATURES = ['Open']#, 'High', 'Low']#, 'Volume']

########## Perceptron CRF ##########
STRUCT_PERCEPTRON_AVG = 200
STRUCT_PERCEPTRON_MAXITER = 30

########## LSTM ##########
input_dim = len(FEATURES)
hidden_dim = 120
num_layers = 2
output_dim = 1
num_epochs = 30
