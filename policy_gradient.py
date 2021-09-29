# Based on https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

from graph_classifier import test
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
from action_space import Actions as A

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    """REINFOCE

    Parameters
    ----------
    in_dim : int
        the input dimension into the neural network
        (Size of the action space)
    hidden_dim: int
        the output dimension of the first lin layer
        the input dimension into the second lin layer
        the output dimension of the second lin layer
    out_dim: int
        the output dimension of the Graph Classification Classifier
    dropout: float
        Probability of neuron dropout after first affine layer
    """

    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(in_dim, hidden_dim)  # 480
        self.dropout = nn.Dropout(p=dropout)
        self.affine2 = nn.Linear(hidden_dim, out_dim)

        self.saved_log_probs = []
        self.rewards = []
        self.to(device)

    def forward(self, x):
        """Return softmax over action space

        Parameters
        ----------
        x : tensor
            action space representation

        Returns
        -------
        tensor
            softmax over action space
        """
        # print(x)
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=0)

    def finish_episode(self, optimizer, eps):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            # print(r)
            R = r + 0.99 * R  # args.gamma = 0.99
            returns.insert(0, R)
        returns = torch.Tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        for param in self.parameters():
            param.grad = None
        policy_loss = torch.cat(policy_loss).to(device).sum()
        policy_loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]


def select_action(state, policy):
    """select action from action set

    Parameters
    ----------
    state : tensor
        associated with class labels of train set

    Returns
    -------
    type: int
        action id
    """
    # extract state (trainset labels = s)
    state = torch.from_numpy(np.array(state)).float().to(device).unsqueeze(0)
    # run through REINFORCE algorithms
    probs = policy(state)
    # transform to Torch Categorical
    m = Categorical(probs)
    # sample from probs
    action = m.sample()
    # save log prob from sampled action to Policy Class
    policy.saved_log_probs.append(m.log_prob(action))
    # Return Action
    return action.item()


def graphs_to_state(G):
    """Return a Tensor of mean node degrees for each graph in G

    Parameters
    ----------
    G : DGL Graph

    Returns
    -------
    Tensor
        mean node degrees for each graph in G
    """
    mean = [torch.Tensor.float(i.in_degrees()).mean() for i in G]  # mean
    maxi = [torch.Tensor.float(i.in_degrees()).max() for i in G]  # max
    mini = [torch.Tensor.float(i.in_degrees()).min() for i in G]  # min
    # assume that the state of the system can be deduced from summary stats
    state = torch.cat((torch.Tensor(mean), torch.Tensor(maxi), torch.Tensor(mini)))
    return state


def perform_action(trainset, action, state, num_classes):
    """Perform action given state, num_classes

    Parameters
    ----------
    action : int
        action id from action selection
    state : tensor
        associated with class labels of train set
    num_classes : int
        number of classes from the init trainset

    Returns
    -------
    tensor
        s': resulting state after taking a in s
    """
    action_size = len(A.action_space)
    # a is the graph label index to change
    a = math.floor(action / action_size)
    # b is the class index to change to
    b = action % action_size

    # perform action on graph
    trainset.graphs[a] = A.action_space[b](trainset.graphs[a])
    # print(f"poisoning graph {a} with label {b}")
    # poison existing label
    state = graphs_to_state(trainset.graphs)
    # return updated state
    # return updated graph
    return state, trainset, a, b
