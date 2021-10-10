import dgl
import networkx as nx
import random


def from_dgl_to_nx(g):
    """converts DGL graph to NX graph

    Parameters
    ----------
    g : graph object

    Returns
    -------
    nx graph
    """
    return g.to_networkx()


def from_nx_to_dgl(g):
    """converts NX graph to DGL graph

    Parameters
    ----------
    g : graph object

    Returns
    -------
    DGL graph
    """
    return dgl.from_networkx(g)


def remove_node(g):
    """remove a node randomly from a graph

    Parameters
    ----------
    g : dgl graph class
    """
    g = from_dgl_to_nx(g)  # convert from dgl to nx
    try:
        node = random.choice(list(g.nodes))  # randomly select node
        g.remove_node(node)  # remove random node
        return from_nx_to_dgl(g)  # convert back to DGL
    except:
        pass


def remove_edge(g):
    """remove an edge randomly from a graph

    Parameters
    ----------
    g : dgl graph class
    """
    try:
        g = from_dgl_to_nx(g)  # convert from dgl to nx
        u, v, _ = random.choice(list(g.edges))
        g.remove_edge(u, v)
        return from_nx_to_dgl(g)  # convert back to DGL
    except:
        pass


def add_node(g):
    """remove a node randomly from a graph

    Parameters
    ----------
    g : dgl graph class
    """
    try:
        g = from_dgl_to_nx(g)  # convert from dgl to nx
        ind = len(g) + 1
        g.add_node(ind)  # remove random node
        return from_nx_to_dgl(g)  # convert back to DGL
    except:
        pass


def add_edge(g):
    """remove an edge randomly from a graph

    Parameters
    ----------
    g : dgl graph class
    """
    try:
        g = from_dgl_to_nx(g)  # convert from dgl to nx
        u, v, _ = random.choice(list(g.edges))
        g.add_edge(u, v)
        return from_nx_to_dgl(g)  # convert back to DGL
    except:
        pass


def do_nothing(g):
    return g


def add_subgraph(g, num_nodes=10, p=0.75):
    """randomly add a gnp graph within the existing graph
    Parameters
    ----------
    g : dgl graph class
    num_nodes: number of nodes in gnp model
    p: probability of edge in gnp model
    """
    g = from_dgl_to_nx(g)  # convert from dgl to nx
    g.update(nx.random_graphs.gnp_random_graph(num_nodes, p))
    return from_nx_to_dgl(g)  # convert back to DGL


class Actions:
    action_space = {
        # 0: remove_edge,
        # # 1: remove_node,
        # 1: add_edge,
        # # 3: add_node,
        # 2: do_nothing,
        0: add_subgraph,
        1: do_nothing,
    }
