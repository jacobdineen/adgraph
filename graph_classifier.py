from dgl.data import MiniGCDataset
import matplotlib.pyplot as plt
import networkx as nx
import dgl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dgl.nn.pytorch import GraphConv
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import config
from torch.cuda.amp.grad_scaler import GradScaler

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# fetch cpu, gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    return batched_graph, torch.Tensor(labels)


class Classifier(nn.Module):
    """Graph Classification Classifier
    Parameters
    ----------
    in_dim : int
        the input dimension into the neural network
    hidden_dim: int
        the output dimension of the first convolutional layer
        the input dimension into the second convolutional layer
        the output dimension of the second convolutional layer
        the input dimension into the linear classifier
    n_classes: int
        the output dimension of the Graph Classification Classifier
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(
            in_dim, hidden_dim, allow_zero_in_degree=True
        )  # first convolutional layer
        self.conv2 = GraphConv(
            hidden_dim, hidden_dim, allow_zero_in_degree=True
        )  # second convolutional layer
        self.classify = nn.Linear(hidden_dim, n_classes)  # linear/output layer
        self.to(device)  # send model to device (cpu/gpu)

    def forward(self, g, intermediate=False):
        """Perform Graph Convolutions on Forward Pass of Classifer
            1) h <- in_degree
            2) h <- Activation(conv1(h))
            3) h <- Activation(conv2(h))
            4) hg <- Average over node representations
            5) y_hat <- Linear(hg)
        Parameters
        ----------
        g : graph
            dgl batch of graphs,label from dataloader
        Returns
        -------
        tensor
            Runs batch through Classifier and outputs scoring function
        """
        # g = dgl.add_self_loop(g)
        g = g.to(device)
        # g = dgl.add_self_loop(g)
        h = g.in_degrees().view(-1, 1).float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g.to(device), h.to(device)))
        h = F.relu(self.conv2(g.to(device), h.to(device)))
        g.ndata["h"] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, "h")
        if intermediate:
            return hg
        else:
            return self.classify(hg)


def train(trainset, epochs: int, model=None):
    """Train GNN on specified trainset
    Parameters
    ----------
    trainset : dataset object
        training data consisting of graph, label pairs
    epochs : int
        number of iterations to train model for
    model : model object
        pretrained torch model
    Returns
    -------
    model object, list
        torch trained model, epoch losses associated with training
    """
    # instantiate data loader object
    data_loader = DataLoader(
        trainset,
        batch_size=config.gcn_batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=True,
    )
    # if there is no model passed in, train as usual else pass model in
    if model is None:
        # instantiate GNN model
        model = Classifier(
            in_dim=1, hidden_dim=config.gcn_hidden, n_classes=trainset.num_classes
        )
        model = model.cuda()
    # instantiate loss function
    loss_func = nn.CrossEntropyLoss()
    # instantiate optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.gcn_learning_rate)
    model.train()  # train mode - learnable params
    epoch_losses = []  # logs of epoch losses
    scaler = GradScaler()
    # Begin Batch Training
    for epoch in range(epochs):
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model(bg.to(device))  # forward pass on data
            loss = loss_func(
                prediction.to(device), label.type(torch.LongTensor).to(device)
            )  # compute loss
            for param in model.parameters():
                param.grad = None
            scaler.scale(loss).backward()  # backward pass
            scaler.step(optimizer)  # perform weight updates
            scaler.update()
            epoch_loss += loss.detach().item()  # compute epoch loss
        epoch_loss /= iter + 1
        epoch_losses.append(epoch_loss)
    return model, epoch_losses


def test(testset, model):
    """return test accuracy given trained model
    Parameters
    ----------
    testset : dataset object
        test set consisting of graph, label pairs
    model : model object
        trained torch model | trainset
    Returns
    -------
    float
        Compute test accuracy as:
            sum(y = y_hat) / len(y)
    """
    test_X, test_Y = map(list, zip(*testset))  # split testset into x,y pairs
    test_bg = dgl.batch(test_X)  # get batch of training instances (X)
    test_Y = torch.tensor(test_Y).float().view(-1, 1).to(device)  # get test labels
    model_Y = model(test_bg)  # run batch X through model
    probs_Y = torch.softmax(model_Y, 1)  # softmax model outputs from probas
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)  # get argmax label (y_hat)

    correct = (test_Y == argmax_Y.float()).sum().item()
    total = len(test_Y)
    return correct / total


def runthrough(trainset, testset, epochs: int):
    """Perform a pass of training/testing given datasets
    Parameters
    ----------
    trainset : dataset class
        training data consisting of graph, label pairs
    testset : dataset class
        training data consisting of graph, label pairs
    epochs : int
        number of iterations to train model for
    Returns
    -------
    model, y
        model: torch model object
        y: torch tensor containing y_hat for test samples from testset
    """
    model, epoch_losses = train(
        trainset=trainset, epochs=epochs, model=None
    )  # run init train on trainset
    model.eval()  # model to eval mode. Freeze params
    y = test(testset, model)  # test given trained params on test set
    return model, y


def poison_test(model, trainset, testset):
    """Retrain Model for single additional epoch with
       poisoned/perturbed data
    Parameters
    ----------
    model : model object
        torch pretrained model
    trainset : dataset object
        train portion of dataset
    testset : dataset object
        test portion of dataset
    Returns
    -------
    tensor
        predictions over test set
    """
    retrained, _ = train(trainset=trainset, epochs=1, model=deepcopy(model))
    retrained = retrained.cuda()
    retrained.eval()
    y = test(testset, retrained)
    return y
