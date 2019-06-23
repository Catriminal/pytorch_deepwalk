#!usr/bin/env python
# -*- coding:utf-8 _*-

from deepwalk import graph
import random


__current_graph = None

def write_walks_to_disk(G, filestr, num_paths, path_length, alpha=0, rand=random.Random(0)):
    global __current_graph
    __current_graph = G
    with open(filestr + '.random_walks', 'w') as fout:
        for walk in graph.build_deepwalk_corpus_iter(G=G, num_paths=num_paths, path_length=path_length, alpha=alpha, rand=rand):
            fout.write(u"{}\n".format(u" ".join(v for v in walk)))
    return filestr + '.random_walks'