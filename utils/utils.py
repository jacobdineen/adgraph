from rl.action_space import add_subgraph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import dgl
from torch import Tensor
import pickle


def save_obj(obj, name):
    with open("data/" + name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def collate(samples):
    """Collate init dataset

    Parameters
    ----------
    samples : tuple
        list of (graph,label) pairs given a dataset

    Returns
    -------
    tuple
        form a mini-batch from a given list of graph and label pairs
    """
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, Tensor(labels)


def perturb_and_visualize(x):
    graph = x
    nx.draw(graph.to_networkx())
    plt.title("original graph")
    plt.show()

    new_graph = add_subgraph(graph, 8)
    nx.draw(new_graph.to_networkx())
    plt.title("Perturbed graph")
    plt.show()


def get_label_mapping(dataset: str = "MiniGC"):
    """fetch label map for respective dataset

    Parameters
    ----------
    dataset : str, optional
        [description], by default 'MiniGC'
    """
    # manual fetch labels from MiniGCDataset
    label_names = [
        "cycle_graph",
        "star_graph",
        "wheel_graph",
        "lollipop_graph",
        "hypercube_graph",
        "grid_graph",
        "complete_graph",
        "circular_ladder_graph",
    ]

    labels = [i for i in range(len(label_names))]
    # TODO add other datasets

    return dict(zip(labels, label_names))


# TODO


def plot_samples(dataset, data, path="images/dataset.png"):
    """return plot showing an example graph/label tuple from each class

    Parameters
    ----------
    data :dgl dataset class
        Currently only support MiniGCDataset
    """

    def plot():
        for i, ax in enumerate(axes.flatten()):
            possible_choices = np.where(data[:][1] == i)[0]
            choice = np.random.choice(possible_choices)
            graph, label = data[choice]
            print(label)
            ax.set_title(
                "Class: {:d}, {}".format(label.item(), label_map[label.item()]), size=18
            )
            nx.draw(graph.to_networkx(), ax=ax)
        plt.savefig(path)
        print(f"saved to {path}")
        plt.show()

    if dataset == "minigc":
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        label_map = get_label_mapping()
        plot()

    if dataset == "imdb" or dataset == "kki" or dataset == "ptc":
        label_map = {0: "negative", 1: "positive"}
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        plot()

    if dataset == "Letter-med":
        classes = [i for i in range(15)]
        labels = [
            "A",
            "E",
            "F",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "T",
            "V",
            "W",
            "X",
            "Y",
            "Z",
        ]
        label_map = dict(zip(classes, labels))
        fig, axes = plt.subplots(3, 5, figsize=(18, 9))
        plot()
