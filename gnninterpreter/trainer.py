import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import copy
import secrets
import os
import pickle
import glob
import torch.nn.functional as F
import torch_geometric as pyg

# TODO: refactor
# from .datasets import *


class Trainer:
    def __init__(self,
                 sampler,
                 discriminator,
                 criterion,
                 scheduler,
                 optimizer,
                 dataset,
                 budget_penalty=None,
                 target_probs: dict[tuple[float, float]] = None,
                 k_samples=32):
        self.k = k_samples
        self.target_probs = target_probs
        self.sampler = sampler
        self.discriminator = discriminator
        self.criterion = criterion
        self.budget_penalty = budget_penalty
        self.scheduler = scheduler
        self.optimizer = optimizer if isinstance(optimizer, list) else [optimizer]
        self.dataset = dataset
        self.iteration = 0

    def probe(self, cls=None, discrete=False):
        graph = self.sampler(k=self.k, discrete=discrete)
        logits = self.discriminator(graph, edge_weight=graph.edge_weight)["logits"].mean(dim=0).tolist()
        if cls is not None:
            return logits[cls]
        return logits

    def detailed_probe(self):
        return pd.DataFrame(dict(
            logits_discrete=(ld := self.probe(discrete=True)),
            logits_continuous=(lc := self.probe(discrete=False)),
            prob_discrete=F.softmax(torch.tensor(ld), dim=0).tolist(),
            prob_continuous=F.softmax(torch.tensor(lc), dim=0).tolist(),
        ))

    def warmup(self, iterations, cls, score):
        orig_criterion = copy.deepcopy(self.criterion)
        orig_iteration = self.iteration
        while self.probe(cls, discrete=True) < score:
            self.criterion = copy.deepcopy(orig_criterion)
            self.iteration = orig_iteration
            self.sampler.init()
            self.train(iterations)

    def train(self, iterations):
        self.bkup_state = copy.deepcopy(self.sampler.state_dict())
        self.bkup_criterion = copy.deepcopy(self.criterion)
        self.bkup_iteration = self.iteration
        self.discriminator.eval()
        self.sampler.train()
        budget_penalty_weight = 1
        for _ in (bar := tqdm(range(int(iterations)), initial=self.iteration, total=self.iteration+iterations)):
            for opt in self.optimizer:
                opt.zero_grad()
            cont_data = self.sampler(k=self.k, mode='continuous')
            disc_data = self.sampler(k=1, mode='discrete', expected=True)
            # TODO: potential bug
            cont_out = self.discriminator(cont_data, edge_weight=cont_data.edge_weight)
            disc_out = self.discriminator(disc_data, edge_weight=disc_data.edge_weight)
            if self.target_probs and all([
                min_p <= disc_out["probs"][0, classes].item() <= max_p
                for classes, (min_p, max_p) in self.target_probs.items()
            ]):
                if self.budget_penalty and self.sampler.expected_m <= self.budget_penalty.budget:
                    break
                budget_penalty_weight *= 1.1
            else:
                budget_penalty_weight *= 0.95

            loss = self.criterion(cont_out | self.sampler.to_dict())
            if self.budget_penalty:
                loss += self.budget_penalty(self.sampler.theta) * budget_penalty_weight
            loss.backward()  # Back-propagate gradients

            # print(self.sampler.omega.grad)

            for opt in self.optimizer:
                opt.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # logging
            size = self.sampler.expected_m
            scores = disc_out["logits"].mean(axis=0).tolist()
            score_dict = {v: scores[k] for k, v in self.dataset.GRAPH_CLS.items()}
            penalty_weight = {'bpw': budget_penalty_weight} if self.budget_penalty else {}
            bar.set_postfix({'size': size} | penalty_weight | score_dict)
            # print(f"{iteration=}, loss={loss.item():.2f}, {size=}, scores={score_dict}")
            self.iteration += 1
        else:
            return False
        return True

    def undo(self):
        self.sampler.load_state_dict(self.bkup_state)
        self.criterion = copy.deepcopy(self.bkup_criterion)
        self.iteration = self.bkup_iteration

    @torch.no_grad()
    def predict(self, G):
        batch = pyg.data.Batch.from_data_list([self.dataset.convert(G, generate_label=True)])
        return self.discriminator(batch)

    @torch.no_grad()
    def quantatitive(self, sample_size=1000, sample_fn=None):
        sample_fn = sample_fn or (lambda: self.evaluate(bernoulli=True))
        p = []
        for i in range(1000):
            p.append(self.predict(sample_fn())["probs"][0].numpy().astype(float))
        return dict(label=list(self.dataset.GRAPH_CLS.values()),
                    mean=np.mean(p, axis=0),
                    std=np.std(p, axis=0))

    @torch.no_grad()
    def quantatitive_baseline(self, **kwargs):
        return self.quantatitive(sample_fn=lambda: nx.gnp_random_graph(n=self.sampler.n, p=1/self.sampler.n),
                                 **kwargs)

    # TODO: do not rely on dataset for drawing
    @torch.no_grad()
    def evaluate(self, *args, show=False, connected=False, **kwargs):
        self.sampler.eval()
        G = self.sampler.sample(*args, **kwargs)
        if connected:
            G = sorted([G.subgraph(c) for c in nx.connected_components(G)], key=lambda g: g.number_of_nodes())[-1]
        if show:
            self.show(G)
            plt.show()
        return G

    def show(self, G, ax=None):
        n = G.number_of_nodes()
        m = G.number_of_edges()
        pred = self.predict(G)
        logits = pred["logits"].mean(dim=0).tolist()
        probs = pred["probs"].mean(dim=0).tolist()
        print(f"{n=} {m=}")
        print(f"{logits=}")
        print(f"{probs=}")
        self.dataset.draw(G, ax=ax)

    def save(self, G, cls_idx, root="result"):
        if isinstance(cls_idx, tuple):
            path = f"{root}/{self.dataset.name}/{self.dataset.GRAPH_CLS[cls_idx[0]]}-{self.dataset.GRAPH_CLS[cls_idx[1]]}"
        else:
            path = f"{root}/{self.dataset.name}/{self.dataset.GRAPH_CLS[cls_idx]}"
        name = secrets.token_hex(4).upper() # TODO: use hash of the graph to avoid duplicate
        os.makedirs(path, exist_ok=True)
        pickle.dump(G, open(f"{path}/{name}.pkl", "wb"))
        self.show(G)
        plt.savefig(f"{path}/{name}.png", bbox_inches="tight")
        plt.show()

    def load(self, id, root="result"):
        path = f"{root}/{self.dataset.name}/*"
        G = pickle.load(open(glob.glob(f"{path}/{id}.pkl")[0], "rb"))
        self.show(G)
        return G

    def evaluate_neg(self, *args, show_neg_edges=True, **kwargs):
        self.sampler.eval()
        neg_edge = self.sampler.sample(*args, **kwargs)
        G = nx.Graph(self.sampler.edges)
        G.remove_edges_from(neg_edge)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        print(f"{n=} {m=}")
        layout = nx.kamada_kawai_layout(G)
        if not show_neg_edges:
            nx.draw(G, pos=layout)
            plt.axis('equal')
            return G
        G.add_edges_from(neg_edge, edge_color='r', width=1)
        edge_color = [G[u][v].get('edge_color', 'k') for u, v in G.edges]
        width = [G[u][v].get('width', 1) for u, v in G.edges]
        nx.draw(G, pos=layout, edge_color=edge_color, width=width)
        plt.axis('equal')
        return G