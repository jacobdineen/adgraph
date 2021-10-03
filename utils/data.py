from dgl.data import MiniGCDataset, TUDataset
from dgl.data.utils import split_dataset
from torch import flatten
import numpy as np
import config

# TODO
# add support for TUDataset


def get_dataset(dataset):
    if dataset == "minigc":
        data = MiniGCDataset(
            config.minigc_size, config.min_graph_nodes, config.max_graph_nodes
        )
        num_classes = 8
    if dataset == "kki":
        data = TUDataset("kki")
        num_classes = 2
    if dataset == "letter_med":
        data = TUDataset("Letter-med")
        num_classes = 15
    if dataset == "imdb":
        data = TUDataset("IMDB-BINARY")
        num_classes = 2

    # reconcile API mismatch here
    if dataset != "minigc":
        data.graphs = data.graph_lists
        data.labels = flatten(data.graph_labels)
        data.num_classes = data.num_labels[0]

    trainset, _, testset = split_dataset(data, [0.8, 0.0, 0.2])

    return trainset.dataset, testset.dataset, num_classes
