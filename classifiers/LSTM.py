import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils.params import TEST_SIZE, SHUFFLE_TRAIN_TEST, batch_size, input_dim, hidden_dim, output_dim, num_layers, num_epochs

import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, hidden_dim). with out - (seq_dim, batch_dim,, hidden_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)  # Bi True?

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


def LSTM_phase(week_features, week_targets):
    print("---------- LSTM Phase -----------")
    print(f"Test size:{TEST_SIZE},"
          f" Shuffle:{SHUFFLE_TRAIN_TEST}, batch size:{batch_size}, "
          f"input_dim:{input_dim}, hidden_dim:{hidden_dim}, output_dim:{output_dim},"
          f" num_layers:{num_layers}, num_epochs:{num_epochs}")

    X_train, X_test, y_train, y_test = train_test_split(week_features,
                                                        week_targets,
                                                        test_size=TEST_SIZE,
                                                        random_state=42,
                                                        shuffle=SHUFFLE_TRAIN_TEST)

    # make training and test sets in torch
    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    X_test = torch.from_numpy(X_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_train, y_train),
                                               batch_size=batch_size,
                                               shuffle=True)

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters())

    print(model)

    loss_list = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        loss_inner_list = []
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs.flatten(), labels.flatten())
            loss.backward()
            optimizer.step()
            loss_inner_list.append(loss.item())
        epoch_mean_loss = np.mean(loss_inner_list)
        print(epoch_mean_loss)
        loss_list.append(epoch_mean_loss)


        model.eval()

        y_train_pred = model(X_test)
        pred = np.round(torch.sigmoid(y_train_pred.detach().squeeze()))

        print("acc:", accuracy_score(y_test.squeeze().flatten(), pred.flatten()))
        model.train()

    plt.plot(loss_list)
    plt.show()
    print('Finished Training')


    """
    
    TODO round input
    
    TODO FC for feature engineers, before
    
    TODO Check if different predictions
    
    TODO add acc to LSTM graph
    
    TODO layers in FC
    
    TODO check 1 week with many zeros
    
    TODO Normalize input (VOLUME)
    
    TODO smooth date (1960-1980)
    
    TODO CrossEntropy..? softmax + NLLLos
    
    CONCAT
    
    Continous input compared to words
    
    FEATURES:
     day before
     2 days before
     day of week
    """

