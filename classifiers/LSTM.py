import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import logging.config

from utils.Params import TEST_SIZE, SHUFFLE_TRAIN_TEST, batch_size, input_dim, hidden_dim, num_layers, \
    num_epochs, output_dim_part1, output_dim_part3, WEIGHTS
from utils.Utils import compute_prediction_report

torch.manual_seed(0)
import matplotlib.pyplot as plt


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        """
        LSTM Neural network , inherits from pytorch nn.Module and a fully connected Linear layer afterwards

        :param input_dim: int, LSTM input dimension - should be equal to number of features used.
        :param hidden_dim: int, size of Hidden dimensions
        :param num_layers: int, Number of hidden layers
        :param output_dim: int, the final output dimension, should be 1
        """
        super(LSTM, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        out, _ = self.lstm(x.to(self.device))
        out = self.fc(out)
        return out


def LSTM_phase(week_features, week_targets, week_targets2=None, is_multilabel=False):
    """
    Trains an LSTM NN on a training set and computes the accuracy on both the training and a validation set

    :param week_features: ndarray, 3d-array of week, day of week, day features
    :param week_targets: ndarray, 2d-array of week, direction of each day
    """
    output_dim = output_dim_part3 if is_multilabel else output_dim_part1

    logger = logging.getLogger(__name__)
    logger.info("---------- LSTM Phase -----------")
    logger.info(f"batch size:{batch_size}, "
                f"input_dim:{input_dim}, hidden_dim:{hidden_dim}, output_dim:{output_dim},"
                f" num_layers:{num_layers}, num_epochs:{num_epochs}")
    is_part1 = week_targets2 is not None

    if week_targets2 is None:
        week_targets2 = week_targets

    X_train, X_test, y_train, y_test, y2_train, y2_test = train_test_split(week_features,
                                                                           week_targets,
                                                                           week_targets2,
                                                                           test_size=TEST_SIZE,
                                                                           random_state=42,
                                                                           shuffle=SHUFFLE_TRAIN_TEST)
    # make training and test sets in torch
    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    X_test = torch.from_numpy(X_test).type(torch.Tensor)
    y_train_torch = torch.from_numpy(y_train).type(torch.int64 if is_multilabel else torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_train, y_train_torch),
                                               batch_size=batch_size,
                                               shuffle=True)
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        logger.debug("using cuda")
        model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(WEIGHTS).to(device)) if is_multilabel else torch.nn.BCEWithLogitsLoss()

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
            outputs = model(inputs.to(device))
            if is_multilabel:
                loss = loss_fn(torch.reshape(input=outputs, shape=(-1, output_dim_part3)).to(device), labels.flatten().to(device))
            else:
                loss = loss_fn(outputs.flatten().to(device), labels.flatten().to(device))
            loss.backward()
            optimizer.step()
            loss_inner_list.append(loss.item())
        epoch_mean_loss = np.mean(loss_inner_list)
        loss_list.append(epoch_mean_loss)

        with torch.no_grad():
            model.eval()
            y_train_pred_prob = model(X_train).cpu()
            y_test_pred_prob = model(X_test).cpu()
            model.train()

            if is_multilabel:
                y_pred_train = np.argmax(y_train_pred_prob, axis=2).detach()
                y_pred_test = np.argmax(y_test_pred_prob, axis=2).detach()
            else:
                y_pred_train = np.round(torch.sigmoid(y_train_pred_prob.detach().squeeze()))
                y_pred_test = np.round(torch.sigmoid(y_test_pred_prob.detach().squeeze()))

            train_acc = accuracy_score(y_train.squeeze().flatten(), y_pred_train.flatten())
            test_acc = accuracy_score(y_test.squeeze().flatten(), y_pred_test.flatten())
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            logger.info(f'epoch={epoch}, train_acc={round(train_acc, 3)}, '
                        f'test_acc={round(test_acc, 3)}, '
                        f'epoch_mean_loss={round(epoch_mean_loss, 3)}')
            compute_prediction_report(y_pred_train.flatten(), y2_train.flatten(), y_train.flatten(), is_part1)
            compute_prediction_report(y_pred_test.flatten(), y2_test.flatten(), y_test.flatten(), is_part1)

    # plt.plot(loss_list)
    # plt.show()
    #
    # plt.plot(range(num_epochs), train_acc_list, range(num_epochs), test_acc_list)
    # plt.show()
