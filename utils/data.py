from dgl.data import MiniGCDataset, TUDataset
import numpy as np

# TODO
# add support for TUDataset


def get_dataset(config):
    trainset = MiniGCDataset(
        config.train_size, config.min_graph_nodes, config.max_graph_nodes
    )
    testset = MiniGCDataset(
        config.test_size, config.min_graph_nodes, config.max_graph_nodes
    )
    num_classes = len(np.unique(trainset.labels))
    return trainset, testset, num_classes
