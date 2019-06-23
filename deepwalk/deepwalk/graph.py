# !/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from six import iterkeys
import random

class Graph(defaultdict):
    def __init__(self):
        super().__init__(list)

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.items()

    def make_undirected(self):
        for v in self.keys():
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        self.make_consistent()
        return self

    def make_consistent(self):
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):
        for x in self:
            if x in self[x]:
                self[x].remove(x)

        return self

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        G = self
        if start:
            path = [start]
        else:
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break

        return [str(node) for node in path]


def load_adjacencylist(file_str, undirected=False, unchecked=True):
    file = open(file_str)
    if unchecked:
        adjlist = parse_adjacencylist_unchecked(file)
        G = convert_from_adjlist_unchecked(adjlist)
    else:
        adjlist = parse_adjacencylist(file)
        G = convert_from_adjlist(adjlist)

    if undirected:
        G = G.make_undirected()

    return G


def parse_adjacencylist(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            introw = [int(x) for x in l.strip().split()]
            row = [introw[0]]
            row.extend(set(sorted(introw[1:])))
            adjlist.extend([row])

    return adjlist


def parse_adjacencylist_unchecked(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            adjlist.extend([[int(x) for x in l.strip().split()]])

    return adjlist


def convert_from_adjlist(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def convert_from_adjlist_unchecked(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0, rand=random.Random(0)):
    nodes = list(G.nodes())
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)



