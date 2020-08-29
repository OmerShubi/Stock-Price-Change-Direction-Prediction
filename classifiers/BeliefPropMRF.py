import numpy as np
from pystruct.models import EdgeFeatureGraphCRF
from utils.Params import FEATURES
import logging.config


def batch_infer(PairsData, StockData, num_days, set_name):
    """
    Performs inference for all companies for all given days by calling _infer function

    :param PairsData: for each company pair, predicted probability for each of the four options (labels),
        for each day in set (ndarray, predictions)
    :param StockData: for each company, both the daily features and label
    :param num_days: int, number of days for inference
    :param set_name: string, 'train'/'test' for logging
    """
    logger = logging.getLogger(__name__)

    logger.info(f"--------- BP for {set_name} Phase ---------")
    stock_to_inx = {stock: inx for inx, stock in enumerate(StockData)}
    num_stocks = len(StockData)
    num_features = len(FEATURES)
    X_batch = np.zeros((num_days, num_stocks, num_features))
    Y_batch = np.zeros((num_days, num_stocks))
    for stock, values in StockData.items():
        # values = (X, y)
        X_batch[:, stock_to_inx[stock]] = values[0]
        Y_batch[:, stock_to_inx[stock]] = values[1].flatten()

    edges = []
    pair_to_inx = {}
    for inx, pair_values in enumerate(PairsData.items()):
        pair, values = pair_values
        edges.append([stock_to_inx[pair[0]], stock_to_inx[pair[1]]])
        pair_to_inx[(pair[0], pair[1])] = 2 * inx
        edges.append([stock_to_inx[pair[1]], stock_to_inx[pair[0]]])
        pair_to_inx[(pair[1], pair[0])] = 2 * inx + 1
    edges = np.array(edges)

    edge_features = np.zeros((num_days, edges.shape[0], num_features))
    for pair, values in PairsData.items():
        # [(0,0), (0,1), (1,0), (1,1)] , [source -> des] : [(Y_source, Y_dest)]
        edge_features[:, pair_to_inx[(pair[0], pair[1])]] = values
        values_inv = values.copy()
        values_inv[2] = values[1]
        values_inv[1] = values[2]
        edge_features[:, pair_to_inx[(pair[1], pair[0])]] = values_inv

    prediction_list = []
    for x, features in zip(X_batch, edge_features):
        prediction_list.append(_infer(x, edges, features))
    prediction_array = np.array(prediction_list)
    for stock, inx in stock_to_inx.items():
        accuracy = (Y_batch[:, inx] == prediction_array[:, inx]).sum() / num_days
        logger.info(f"BeliefProp accuracy {set_name} for {stock} : {round(accuracy, 3)}")


def _infer(nodes_features, edges, edge_features):
    """
    given the graph node features (companies and their daily features),
     edges (for related companies) and edge_features (weights of connections)
    predicts the direction of stock price change for each of the companies

    :return: prediction for each of the companies
    """
    # simple example
    # nodes_features = np.array([[4], [1]])
    # Y = np.array([1, 0])
    # edges = np.array([[0, 1], [1, 0]])
    # [(0,0), (0,1), (1,0), (1,1)]
    # [source -> des] : [(Y_source, Y_dest)]
    # edge_features = np.array([[0, 1000, 0, 0], [0, 0, 1000, 0]])

    n_states = 2  # num of classes
    n_features = nodes_features[0].shape[0]
    n_edge_features = n_states * n_states
    crf = EdgeFeatureGraphCRF(n_states=n_states,
                              n_features=n_features,
                              n_edge_features=n_edge_features,
                              inference_method='max-product')

    X = (nodes_features.reshape((-1, n_features)), edges, edge_features)
    # w.shape = n_states*n_features + n_states*n_states*n_edge_features
    w = np.hstack([np.ones((n_states * n_features)).ravel(), np.eye((n_edge_features)).ravel()])
    Y_pred = crf.inference(X, w)

    return Y_pred
