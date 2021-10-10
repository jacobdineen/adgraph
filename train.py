import numpy as np
import config

from graph_classifier import runthrough, poison_test
from rl.policy_gradient import Policy, select_action, perform_action, graphs_to_state
from utils.data import get_dataset
from utils.utils import save_obj

import torch
import torch.optim as optim

import pickle
from rl.action_space import Actions as A
import logging


torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# TODO


def set_policy(trainset, A):
    policy = Policy(
        in_dim=len(graphs_to_state(trainset.graphs)),  # num graphs
        hidden_dim=256,  # num hidden neurons
        out_dim=len(trainset) * 2,  # num possible actions
        dropout=0.6,
    )

    optimizer = optim.Adam(policy.parameters(), lr=config.rl_learning_rate)
    return policy, optimizer, config.eps


def trn(
    g_model,
    init_reward,
    trainset,
    testset,
    num_epochs: int,
    attacking_budget: float,
    run_num: int,
    num_classes: int,
    dataset: str,
):

    points = int(len(trainset) * attacking_budget)
    pickle.dump(trainset.graphs, open("data/init_graphs.p", "wb"))

    policy, optimizer, eps = set_policy(trainset, A)

    for i_episode in range(num_epochs):
        # after each attempt of poisoning via label, revert to original labels
        # unpickle original graph structure
        trainset.graphs = pickle.load(open("data/init_graphs.p", "rb"))
        state, ep_reward, _ = graphs_to_state(trainset.graphs), 0, 0
        # # For first poisoning point, reward is how much better ("worse") than baseline acc.
        prev_acc = init_reward
        episode_actions = []
        graph_classes = []
        for t in range(points):  # Assume 18 Poisoning points
            # fetch action | state
            action = select_action(state, policy)
            # move to s' given a in s
            state, trainset, graph, action, graph_class = perform_action(
                trainset, action, state, num_classes
            )
            episode_actions.append(action)
            graph_classes.append(graph_class)

            # get acc on poison test with modified data
            curr_acc = poison_test(model=g_model, trainset=trainset, testset=testset)

            reward = prev_acc - curr_acc

            prev_acc = curr_acc
            policy.rewards.append(reward)
            ep_reward += reward

        # perform policy updates/optimization
        policy.finish_episode(optimizer, eps)
        logging.info(
            f"rl; {dataset}; {attacking_budget}; {run_num};  {i_episode}; {init_reward:.4f}; {curr_acc:.4f}; {ep_reward:.4f}; {graph_classes};  {episode_actions}"
        )


def trn_random(
    g_model,
    init_reward,
    trainset,
    testset,
    num_epochs: int,
    attacking_budget: float,
    run_num: int,
    num_classes: int,
    dataset: str,
):
    points = int(len(trainset) * attacking_budget)
    pickle.dump(trainset.graphs, open(config.data_save_path, "wb"))

    for i_episode in range(num_epochs):
        # reload init graphs
        trainset.graphs = pickle.load(open(config.data_save_path, "rb"))
        state, ep_reward, _ = graphs_to_state(trainset.graphs), 0, 0
        prev_acc = init_reward

        episode_actions = []
        graph_classes = []
        for t in range(points):  # Assume 18 Poisoning points
            action = np.random.choice(len(trainset) * len(A.action_space))
            try:
                state, trainset, graph, action, graph_class = perform_action(
                    trainset, action, state, num_classes
                )

                episode_actions.append(action)
                graph_classes.append(graph_class)

                curr_acc = poison_test(
                    model=g_model, trainset=trainset, testset=testset
                )

                reward = prev_acc - curr_acc
                prev_acc = curr_acc
                ep_reward += reward
            except:
                pass

        logging.info(
            f"rand; {dataset}; {attacking_budget}; {run_num}; {i_episode}; {init_reward:.4f}; {curr_acc:.4f}; {ep_reward:.4f}; {graph_classes}; {episode_actions}"
        )


if __name__ == "__main__":
    logging.basicConfig(filename="data/data.log", level=logging.INFO)
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(
        f"METHOD; DATASET; POISON_POINTS; RUN_NUMBER; EPISODE; INIT_REWARD; CURRENT_ACCURACY; EPISODIC_REWARD; GRAPH_CLASSES_PERTURBED; EPISODE_ACTIONS"
    )
    for ind, dataset in enumerate(config.datasets):
        trainset, testset, num_classes = get_dataset(dataset)

        g_model, init_reward = runthrough(
            trainset=trainset,
            testset=testset,
            epochs=config.num_graph_epochs,
        )
        for run_num in range(config.num_runs):
            for budget in config.attacking_budget:
                for method in ["policy", "random"]:
                    print(
                        f"Dataset: {dataset} | Run: {run_num} | Budget: {budget} | Method: {method}"
                    )

                    if method == "random" and run_num < 1:
                        trn_random(
                            g_model=g_model,
                            init_reward=init_reward,
                            trainset=trainset,
                            testset=testset,
                            num_epochs=config.num_epochs,
                            attacking_budget=budget,
                            run_num=run_num,
                            num_classes=num_classes,
                            dataset=dataset,
                        )

                    if method == "policy":
                        trn(
                            g_model=g_model,
                            init_reward=init_reward,
                            trainset=trainset,
                            testset=testset,
                            num_epochs=config.num_epochs,
                            attacking_budget=budget,
                            run_num=run_num,
                            num_classes=num_classes,
                            dataset=dataset,
                        )
