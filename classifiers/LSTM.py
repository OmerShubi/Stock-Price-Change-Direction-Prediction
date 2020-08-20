import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import logging.config

from utils.Params import TEST_SIZE, SHUFFLE_TRAIN_TEST, batch_size, input_dim, hidden_dim, output_dim, num_layers, \
    num_epochs, FEATURES

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
    logger = logging.getLogger(__name__)
    logger.info("---------- LSTM Phase -----------")
    logger.info(f"batch size:{batch_size}, "
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

    loss_list = []
    train_acc_list = []
    test_acc_list = []
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
        loss_list.append(epoch_mean_loss)

        with torch.no_grad():
            model.eval()
            y_train_pred = model(X_train)
            y_test_pred = model(X_test)
            model.train()

            y_pred_train = np.round(torch.sigmoid(y_train_pred.detach().squeeze()))
            y_pred_test = np.round(torch.sigmoid(y_test_pred.detach().squeeze()))

            train_acc = accuracy_score(y_train.squeeze().flatten(), y_pred_train.flatten())
            test_acc = accuracy_score(y_test.squeeze().flatten(), y_pred_test.flatten())
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

        if epoch % 10 or epoch == num_epochs - 1:
            logger.info(f'epoch={epoch}, train_acc={round(train_acc, 3)}, '
                        f'test_acc={round(test_acc, 3)}, '
                        f'epoch_mean_loss={round(epoch_mean_loss, 3)}')

    # plt.plot(loss_list)
    # plt.show()
    #
    # plt.plot(range(num_epochs), train_acc_list, range(num_epochs), test_acc_list)
    # plt.show()
