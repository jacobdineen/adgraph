from dgl.data import MiniGCDataset, TUDataset


def get_dataset(dataset, train_size, test_size):
    if dataset == "mingc":
        min_graph_nodes = 1500
        max_graph_nodes = 2000
        return (
            MiniGCDataset(train_size, min_graph_nodes, max_graph_nodes),
            MiniGCDataset(test_size, min_graph_nodes, max_graph_nodes),
            8,
        )
