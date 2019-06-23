#! /usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from deepwalk import graph
from deepwalk import walks
from deepwalk.word2vec import Word2Vec
import random

def process(args):
    if args.format == "adjlist":
        G = graph.load_adjacencylist(args.input, undirected=args.undirected)
    elif args.format == "edgelist":
        G = graph.load_adjacencylist(args.input, undirected=args.undirected)
    else:
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist'" % args.format)

    print("Number of nodes: {}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * args.number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * args.walk_length

    print("Data size (walks*length): {}".format(data_size))

    print("Walking...")
    walk_file = walks.write_walks_to_disk(G, args.output, num_paths=args.number_walks,
                                            path_length=args.walk_length, alpha=0,
                                          rand=random.Random(args.seed))
    model = Word2Vec(walk_file, args.output, emb_dimension=args.representation_size, window_size=args.window_size,
                     min_count=0)
    print("Training...")
    model.skip_gram_train()



def main():
    parser: ArgumentParser = ArgumentParser("deepwalk", formatter_class=ArgumentDefaultsHelpFormatter,
                                            conflict_handler='resolve')
    parser.add_argument('--format', default='adjlist', choices=['adjlist', 'edgelist'])
    parser.add_argument('--number-walks', default=10, type=int)
    parser.add_argument('--input', nargs='?', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--representation-size', default=64, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')
    parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                        help='Use vertex degree to estimate the frequency of nodes '
                             'in the random walks. This option is faster than '
                             'calculating the vocabulary.')
    parser.add_argument('--walk-length', default=40, type=int)
    parser.add_argument('--window-size', default=5, type=int,
                        help='Window size of skipgram model.')

    # args = parser.parse_args()
    args = parser.parse_args("--input ./example_graphs/karate.adjlist "
                             "--output ./output "
                             "--representation-size 2".split())

    print(args)
    process(args)


if __name__ == "__main__":
    main()
