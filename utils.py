import dgl
from dgl.data import MiniGCDataset
from action_space import add_subgraph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


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


def plot_samples(data, path="images/dataset.png"):
    """return plot showing an example graph/label tuple from each class

    Parameters
    ----------
    data :dgl dataset class
        Currently only support MiniGCDataset
    """
    fig, axes = plt.subplots(2, 4, figsize=(12, 9))
    label_map = get_label_mapping()
    for i, ax in enumerate(axes.flatten()):
        possible_choices = np.where(data[:][1] == i)[0]
        choice = np.random.choice(possible_choices)
        graph, label = data[choice]
        ax.set_title("Class: {:d}, {}".format(label, label_map[label.item()]))
        nx.draw(graph.to_networkx(), ax=ax)
    plt.savefig(path)
    print(f"saved to {path}")
    plt.show()
