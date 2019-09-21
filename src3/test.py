import math
import numpy as np
import networkx as nx
import node2vec_ms
import node2vec_ms_walk
import node2vec_ms_walk2
import node2vec_revers
import node2vec
from gensim.models import Word2Vec
import time
from distance import get_matrixs, calc_matrix_norm, cluster_distance

config = {
    # 'input': '../graph/facebook_combined.edgelist',
    # 'input': '../graph/artist_edges.edgelist',
    'input': '../graph/email-Eu-core.txt',
    # 'input': '../graph/lesmis.edgelist',
    # 'output': '../emb/lesmis{}.emb',
    'dimensions': 16,
    'walk_length': 80,
    'num_walks': 120,
    'window_size': 10,
    'iter': 10,
    'workers': 8,
    'p': 1,
    'q': 1,
    'weighted': False,
    'directed': False,
    'unweighted': True

}


def read_graph(config):
    '''
    Reads the input network in networkx.
    '''
    if config['weighted']:
        G = nx.read_edgelist(config['input'], nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(config['input'], nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not config['directed']:
        G = G.to_undirected()

    return G


def learn_embeddings(walks, config):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=config['dimensions'], window=config['window_size'], min_count=0, sg=1,
                     workers=config['workers'], iter=config['iter'])

    model.wv.save_word2vec_format(config['output'])

    return list(zip(model.wv.vectors, model.wv.index2entity))


def test(config, impl, nc, log_stats=False):
    start = time.time()
    nx_G = read_graph(config)
    G = impl.Graph(nx_G, config['directed'], config['p'], config['q'], log_stats)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(config['num_walks'], config['walk_length'], nc)
    walk_end = time.time()
    print(f"Walk Time: {walk_end - start}, Concurrent walks: {nc}")
    # emb =learn_embeddings(walks, config)
    # print(f"Emb Time: {time.time() - walk_end}")
    # print(f"Total Time: {time.time() - start}")
    # return emb
    return 1


test_count = 1

output_schema = '../emb/lesmis{}.emb',

outputs_ms = [('../emb/lesmis' + str(i + 1) + '.emb', "MS_" + str(i + 1)) for i in range(test_count)]
outputs_base = [('../emb/lesmis' + str(i + 1) + '.emb', "base_" + str(i + 1)) for i in
                range(test_count, 2 * test_count)]

res = {}
print("Base times")
for el in [1]:
    test(config, node2vec, el)
# #
print()
print("No hash grouping")
for el in [1]:
    test(config, node2vec_ms, el)

print()
print("Hash grouping")
# for el in [1, 4, 8, 16, 32, 64, 128]:
for el in [1]:
    test(config, node2vec_ms_walk, el)

# print()
# print("Hash grouping2")
# for el in [1, 2, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512]:
#     test(config, node2vec_ms_walk2, el)

# for i in range(10):
#     test(config, node2vec_ms, 2**(i+1))

# test(config, node2vec_ms, 4)
#
# for i in range(test_count):
#     config['output'] = f"../emb/lesmis_ms{i+1}.emb"
#     res[f'lesmis_ms{i+1}'] = test(config, node2vec_ms, 1)

# for i in range(test_count, 2 * test_count):
#     config['output'] = '../emb/lesmis' + str(i + 1) + '.emb'
#     res.append(test(config, node2vec_revers))

# for i in range(test_count):
#     config['output'] = f"../emb/lesmis{i+1}.emb"
#     res[f'lesmis{i+1}'] = test(config, node2vec, 4)

# cluster_distance(res, 6)

# calc_matrix_norm(get_matrixs('../emb/lesmis{}.emb', 3*test_count))
