#!/usr/bin/env python3
"""
End-to-end pipeline for RL-based graph isomorphism experiments.

This script trains the model, evaluates it on synthetic and TU datasets,
compares against VF2 on hard graphs, and saves all artifacts under outputs/.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.datasets import TUDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pyg_to_nx(data):
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        G.add_edge(u, v)
    # Ensure isolated nodes are not lost.
    G.add_nodes_from(range(data.num_nodes))
    return nx.convert_node_labels_to_integers(G)


def wl_colors(G, iters=3):
    colors = {n: G.degree[n] for n in G.nodes()}
    for _ in range(iters):
        new_colors = {}
        for node in G.nodes():
            neigh = sorted(colors[n] for n in G.neighbors(node))
            new_colors[node] = hash((colors[node], tuple(neigh)))
        mapping = {c: i for i, c in enumerate(set(new_colors.values()))}
        colors = {n: mapping[c] for n, c in new_colors.items()}
    return colors


def hard_non_iso(G):
    G2 = G.copy()
    edges = list(G2.edges())

    for _ in range(max(3, len(edges) // 10)):
        if len(edges) < 2:
            break

        (u1, v1), (u2, v2) = random.sample(edges, 2)

        if len({u1, v1, u2, v2}) < 4:
            continue
        if G2.has_edge(u1, v2) or G2.has_edge(u2, v1):
            continue

        if G2.has_edge(u1, v1):
            G2.remove_edge(u1, v1)
        if G2.has_edge(u2, v2):
            G2.remove_edge(u2, v2)

        G2.add_edge(u1, v2)
        G2.add_edge(u2, v1)
        edges = list(G2.edges())

    return G2


def generate_pair(n_range=(20, 60)):
    n = random.randint(*n_range)
    p = max(0.1, min(0.3, 5.0 / n))
    G1 = nx.fast_gnp_random_graph(n, p)

    if random.random() < 0.5:
        perm = list(range(n))
        random.shuffle(perm)
        G2 = nx.relabel_nodes(G1, dict(enumerate(perm)))
        return G1, G2, True
    return G1, hard_non_iso(G1), False


def generate_real_pair_ds(dataset):
    G1 = pyg_to_nx(random.choice(dataset))
    n = G1.number_of_nodes()

    if n < 5:
        return generate_real_pair_ds(dataset)

    if random.random() < 0.5:
        perm = list(range(n))
        random.shuffle(perm)
        G2 = nx.relabel_nodes(G1, dict(enumerate(perm)))
        return G1, G2, True
    return G1, hard_non_iso(G1), False


class Env:
    def reset(self, G1, G2, label):
        assert G1.number_of_nodes() == G2.number_of_nodes()

        self.G1, self.G2 = G1, G2
        self.label = label
        self.n = G1.number_of_nodes()
        self.mapping = {}
        self.used_v = set()

        self.tri1 = nx.triangles(G1)
        self.tri2 = nx.triangles(G2)
        self.clust1 = nx.clustering(G1)
        self.clust2 = nx.clustering(G2)
        self.wl1 = wl_colors(G1)
        self.wl2 = wl_colors(G2)
        return self

    def select_node(self):
        remaining = [x for x in range(self.n) if x not in self.mapping]
        return max(remaining, key=lambda x: self.G1.degree[x])

    def valid_actions(self, u):
        u_deg = self.G1.degree[u]
        u_tri = self.tri1[u]
        return [
            v for v in range(self.n)
            if v not in self.used_v
            and abs(self.G2.degree[v] - u_deg) <= 1
            and abs(self.tri2[v] - u_tri) <= 2
        ]

    def step(self, u, v):
        if not self.check_valid(u, v):
            return -8.0, True

        self.mapping[u] = v
        self.used_v.add(v)

        if len(self.mapping) == self.n:
            return (+25.0 if self.label else -25.0), True

        return +1.0, False

    def check_valid(self, u, v):
        u_nbrs = set(self.G1.neighbors(u))
        for u_old, v_old in self.mapping.items():
            if (u_old in u_nbrs) != self.G2.has_edge(v, v_old):
                return False
        return True


def extract_features(env, u, v_list):
    feats = []
    n = env.n

    u_deg = env.G1.degree[u]
    u_tri = env.tri1[u]
    u_cl = env.clust1[u]
    u_wl = env.wl1[u]
    u_nbrs = list(env.G1.neighbors(u))
    u_2hop = sum(env.G1.degree[nbr] for nbr in u_nbrs)

    max_wl = max(max(env.wl1.values()), max(env.wl2.values())) + 1

    for v in v_list:
        v_deg = env.G2.degree[v]
        v_tri = env.tri2[v]
        v_cl = env.clust2[v]
        v_wl = env.wl2[v]
        v_2hop = sum(env.G2.degree[nbr] for nbr in env.G2.neighbors(v))

        consistency = 1.0
        if env.mapping:
            matches = sum(
                (u_o in u_nbrs) == env.G2.has_edge(v, v_o)
                for u_o, v_o in env.mapping.items()
            )
            consistency = matches / len(env.mapping)

        feats.append([
            abs(u_deg - v_deg) / max(n, 1),
            abs(u_tri - v_tri) / max(n, 1),
            abs(u_cl - v_cl),
            consistency,
            len(env.mapping) / max(n, 1),
            u_deg / max(n, 1),
            v_deg / max(n, 1),
            abs(u_wl - v_wl) / max(max_wl, 1),
            abs(u_2hop - v_2hop) / max(n * n, 1),
        ])

    return torch.tensor(feats, dtype=torch.float32)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, 1)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.net(x)
        return self.actor(h), self.critic(h)


def train(model, optimizer, mutag_dataset, episodes, gamma):
    model.train()
    losses = []

    for ep in tqdm(range(episodes), desc="Training"):
        if random.random() < 0.7:
            G1, G2, label = generate_pair()
        else:
            G1, G2, label = generate_real_pair_ds(mutag_dataset)

        env = Env().reset(G1, G2, label)
        log_probs, values, rewards = [], [], []
        done = False

        while not done:
            u = env.select_node()
            actions = env.valid_actions(u)

            if not actions:
                rewards.append(-5.0)
                break

            feats = extract_features(env, u, actions)
            logits, val = model(feats)
            probs = torch.softmax(logits.view(-1), dim=0)

            dist = torch.distributions.Categorical(probs)
            idx = dist.sample()

            log_probs.append(dist.log_prob(idx))
            values.append(val.view(-1)[idx])

            v = actions[idx]
            reward, done = env.step(u, v)
            rewards.append(reward)

        if not values:
            continue

        returns, G = [], 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        T = min(len(returns), len(values))
        returns = torch.tensor(returns[:T])
        values = torch.stack(values[:T])
        log_probs = torch.stack(log_probs[:T])

        adv = returns - values.detach()
        loss = -(log_probs * adv).mean() + 0.5 * adv.pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return losses


def run_policy(model, G1, G2, label):
    env = Env().reset(G1, G2, label)
    success = True
    done = False

    while not done:
        u = env.select_node()
        actions = env.valid_actions(u)

        if not actions:
            success = False
            break

        feats = extract_features(env, u, actions)
        with torch.no_grad():
            logits, _ = model(feats)
            idx = torch.argmax(logits.view(-1)).item()

        v = actions[idx]
        reward, done = env.step(u, v)

        if reward < 0:
            success = False

    return success


def evaluate_generator(model, name, generator, trials):
    correct = 0
    start = time.time()

    for _ in tqdm(range(trials), desc=f"Eval {name}"):
        G1, G2, label = generator()
        pred = run_policy(model, G1, G2, label)
        if pred == label:
            correct += 1

    return {
        "dataset": name,
        "trials": trials,
        "accuracy": correct / trials,
        "avg_time_sec": (time.time() - start) / trials,
    }


def evaluate_dataset(model, name, dataset, trials):
    return evaluate_generator(
        model,
        name,
        lambda: generate_real_pair_ds(dataset),
        trials,
    )


def generate_regular_graph(n, k=4):
    if (n * k) % 2 != 0:
        k -= 1
    return nx.random_regular_graph(k, n)


def generate_grid_graph(n):
    side = int(n**0.5)
    G = nx.grid_2d_graph(side, side)
    return nx.convert_node_labels_to_integers(G)


def generate_cycle_graph(n):
    return nx.cycle_graph(n)


def hard_graph_pair(n=50):
    choice = random.choice(["regular", "grid", "cycle"])

    if choice == "regular":
        G1 = generate_regular_graph(n, k=4)
    elif choice == "grid":
        G1 = generate_grid_graph(n)
    else:
        G1 = generate_cycle_graph(n)

    if random.random() < 0.5:
        perm = list(range(len(G1)))
        random.shuffle(perm)
        G2 = nx.relabel_nodes(G1, dict(enumerate(perm)))
        return G1, G2, True

    G2 = G1.copy()
    edges = list(G2.edges())

    if len(edges) >= 2:
        (u1, v1), (u2, v2) = random.sample(edges, 2)
        if len({u1, v1, u2, v2}) == 4:
            if G2.has_edge(u1, v1):
                G2.remove_edge(u1, v1)
            if G2.has_edge(u2, v2):
                G2.remove_edge(u2, v2)
            G2.add_edge(u1, v2)
            G2.add_edge(u2, v1)

    return G1, G2, False


def evaluate_hard_with_vf2(model, n, trials):
    rl_correct = 0
    vf2_correct = 0
    rl_time = 0.0
    vf2_time = 0.0

    for _ in tqdm(range(trials), desc=f"Hard n={n}"):
        G1, G2, label = hard_graph_pair(n)

        start = time.time()
        rl_pred = run_policy(model, G1, G2, label)
        rl_time += time.time() - start
        if rl_pred == label:
            rl_correct += 1

        start = time.time()
        matcher = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
        vf2_pred = matcher.is_isomorphic()
        vf2_time += time.time() - start
        if vf2_pred == label:
            vf2_correct += 1

    return {
        "dataset": f"hard_graph_n_{n}",
        "trials": trials,
        "rl_accuracy": rl_correct / trials,
        "vf2_accuracy": vf2_correct / trials,
        "rl_avg_time_sec": rl_time / trials,
        "vf2_avg_time_sec": vf2_time / trials,
    }


def save_csv(path, rows):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output_dir, losses, eval_rows, hard_rows):
    import matplotlib.pyplot as plt

    if eval_rows:
        plt.figure()
        names = [row["dataset"] for row in eval_rows]
        values = [row["accuracy"] for row in eval_rows]
        plt.bar(names, values)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Evaluation accuracy")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "evaluation_accuracy.png")
        plt.close()

    if hard_rows:
        plt.figure()
        names = [row["dataset"] for row in hard_rows]
        rl_values = [row["rl_accuracy"] for row in hard_rows]
        vf2_values = [row["vf2_accuracy"] for row in hard_rows]
        x = np.arange(len(names))
        width = 0.35
        plt.bar(x - width / 2, rl_values, width, label="RL")
        plt.bar(x + width / 2, vf2_values, width, label="VF2")
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("RL vs VF2 hard graph accuracy")
        plt.xticks(x, names, rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "rl_vs_vf2_accuracy.png")
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Relative path for downloaded datasets")
    parser.add_argument("--output-dir", default="outputs", help="Relative path for artifacts")
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes")
    parser.add_argument("--trials", type=int, default=50, help="Evaluation trials per dataset")
    parser.add_argument("--hard-trials", type=int, default=30, help="Trials for hard graph VF2 comparison")
    parser.add_argument("--hard-sizes", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print("Loading datasets...")
    datasets = {
        "MUTAG": TUDataset(root=str(data_dir / "MUTAG"), name="MUTAG"),
        "PROTEINS": TUDataset(root=str(data_dir / "PROTEINS"), name="PROTEINS"),
        "IMDB-BINARY": TUDataset(root=str(data_dir / "IMDB"), name="IMDB-BINARY"),
    }

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Training model...")
    losses = train(model, optimizer, datasets["MUTAG"], args.episodes, gamma=0.97)

    model_path = output_dir / "actor_critic_model.pt"
    torch.save(model.state_dict(), model_path)

    print("Evaluating model...")
    eval_rows = [
        evaluate_generator(model, "synthetic", generate_pair, args.trials),
        evaluate_dataset(model, "MUTAG", datasets["MUTAG"], args.trials),
        evaluate_dataset(model, "PROTEINS", datasets["PROTEINS"], args.trials),
        evaluate_dataset(model, "IMDB-BINARY", datasets["IMDB-BINARY"], args.trials),
    ]

    hard_rows = [
        evaluate_hard_with_vf2(model, n=size, trials=args.hard_trials)
        for size in args.hard_sizes
    ]

    results = {
        "config": vars(args),
        "evaluation": eval_rows,
        "hard_graphs": hard_rows,
        "artifacts": {
            "model": str(model_path),
            "evaluation_csv": str(output_dir / "evaluation_results.csv"),
            "hard_graph_csv": str(output_dir / "hard_graph_results.csv"),
            "evaluation_accuracy_plot": str(output_dir / "evaluation_accuracy.png"),
            "rl_vs_vf2_plot": str(output_dir / "rl_vs_vf2_accuracy.png"),
        },
    }

    save_csv(output_dir / "evaluation_results.csv", eval_rows)
    save_csv(output_dir / "hard_graph_results.csv", hard_rows)

    with (output_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    with (output_dir / "training_log.json").open("w") as f:
        json.dump({"losses": losses}, f)

    save_plots(output_dir, losses, eval_rows, hard_rows)

    print("\nPipeline completed successfully.")
    print(f"Artifacts saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
