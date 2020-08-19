import numpy as np
from pystruct.models import EdgeFeatureGraphCRF


def infer(nodes_features, Y):
    # nodes_features = np.array([[4], [1]])
    # Y = np.array([1, 0])
    edges = np.array([[0, 1], [1, 0]])
    # [(0,0), (0,1), (1,0), (1,1)]
    # [source -> des] : [(Y_source, Y_dest)]
    edge_features = np.array([[0, 1000, 0, 0], [0, 100, 0, 0]])

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
    # batch_inference
    Y_pred = crf.inference(X, w)

    accuracy = calc_accuracy(Y_pred, Y)
    print(accuracy)

def calc_accuracy(Y_pred, Y_true):
    pass