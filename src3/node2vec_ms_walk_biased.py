import numpy as np
import random
from itertools import islice, chain
import json
import pickle


class WalkAggregator:

    def __init__(self, current: int, walks: list, walks_length: int = 1, prev: int = None):
        self.current = current
        self.prev = prev
        self.walks_length = walks_length
        self.walks_count = len(walks)
        self.walks = walks

    def inc_len(self):
        self.walks_length += 1

    def __hash__(self) -> int:
        return hash(self.current) if self.prev is None else hash((self.current, self.prev))

    def __eq__(self, other) -> bool:
        return self.__class__ == other.__class__ and self.current == other.current and self.prev == other.prev


class Graph:

    def __init__(self, nx_G, is_directed, p, q, log_stats=False):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.neighbors = {}
        self.edges_count = {}
        self.start_nodes = {}
        for node in self.G.nodes:
            self.neighbors[node] = sorted(self.G.neighbors(node))
        self.log_stats = log_stats
        self.three_in_row = {}
        self.four_in_row = {}

    def neighbor_chunker(self, n):
        neighbors = set(self.neighbors.keys())
        while True:
            chunk = []
            while True:
                if len(chunk) == n or len(neighbors) == 0:
                    break
                node = neighbors.pop()
                chunk.append(node)
                if len(chunk) == n or len(neighbors) == 0:
                    break
                for no in self.neighbors[node]:
                    try:
                        neighbors.remove(no)
                    except Exception:
                        if len(chunk) == n or len(neighbors) == 0:
                            break
                    else:
                        chunk.append(no)
                        if len(chunk) == n or len(neighbors) == 0:
                            break
            if len(neighbors) == 0 and len(chunk) == 0:
                return
            yield iter(chunk)

    def chunker(self, iterable, n):
        it = iter(iterable)
        while True:
            chunk = islice(it, n)
            try:
                first = next(chunk)
            except StopIteration:
                return
            yield chain((first,), chunk)

    def draw_node(self, node, steps_number, node_neighbors):
        result = []

        n1 = self.alias_nodes[node][0]
        n2 = self.alias_nodes[node][1]
        for _ in range(steps_number):
            alias = alias_draw(n1, n2)
            next = node_neighbors[alias]
            result.append(next)
        return result

    def draw_edge(self, pair, steps_number, node_neighbors, saved_step, reuse_probability):
        result = []
        n1, n2 = self.alias_edges[(pair[1], pair[0])]
        for _ in range(steps_number):
            if saved_step.get(pair[0]) and reuse_probability > random.random():
                next = saved_step[pair[0]]
            else:
                alias = alias_draw(n1, n2)
                next = node_neighbors[alias]
                saved_step[pair[0]] = next
            result.append(next)
        return result

    def update_step(self, drawn, current_node, walks, visit_dict_next, completed_walks, walk_len):
        # np.random.shuffle(drawn) consider if this is necessary (or some other way of randomizing the order).
        # Simply iterating drawn might introduce some biasp
        for new_node in drawn:
            new_pair = new_node, current_node
            new_walk = walks.pop() + [new_node]
            if len(new_walk) == walk_len:
                completed_walks.append(new_walk)
            else:
                visit_dict_next[new_pair] = visit_dict_next.get(new_pair, [])
                visit_dict_next[new_pair].append(new_walk)

    def get_prev(self, walk):
        return walk[-2] if len(walk) > 1 else -1

    def node2vec_walk(self, walk_length, start_nodes, num_walks, batch_id, reuse_probability):
        '''
        Simulate a random walk starting from start node.
        '''

        visit_dict = {}
        visit_dict_next = {}
        saved_step = {}
        completed_walks = []
        G = self.G

        for start_node in start_nodes:
            visit_dict[(start_node, None)] = [[start_node] for _ in range(num_walks)]
        while visit_dict:
            while visit_dict:
                (current_node, previous_node), walks = visit_dict.popitem()
                cur_nbrs = self.neighbors[current_node]
                if self.log_stats:
                    self.update_edges_count(current_node, previous_node, len(walks), len(walks[0]), walks[0][0],
                                            walks[0],
                                            batch_id)
                if len(cur_nbrs) > 0:
                    if len(walks[0]) == 1:
                        drawn = self.draw_node(current_node, len(walks), cur_nbrs)
                        self.update_step(drawn, current_node, walks, visit_dict_next, completed_walks, walk_length)
                    else:
                        drawn = self.draw_edge((current_node, previous_node), len(walks), cur_nbrs, saved_step,
                                               reuse_probability)
                        self.update_step(drawn, current_node, walks, visit_dict_next, completed_walks, walk_length)
                else:
                    print("HANDLE NODE WITH NO NEIGHBORS")  # Probably just continue or add those walks to result
            visit_dict, visit_dict_next = visit_dict_next, visit_dict

        return completed_walks

    def update_edges_count(self, cur, prev, count, step, starting_node, walk, batch_id):
        key = str((cur, prev))
        cur_count = self.edges_count.get(key)
        cur_start = self.start_nodes.get(key)
        if not cur_count:
            self.edges_count[key] = {
                "c": count,  # count
                "sn": len(self.neighbors[cur]),  # source nodes
                "tn": len(self.neighbors[prev]) if self.neighbors.get(prev) else 0,  # target nodes
                "ta": {step: [count]}  # time access
                # "b": {batch_id: 1}  # batch id
            }
            self.start_nodes[key] = {
                starting_node: 1  # starting nodes
            }
        else:
            cur_count['c'] += count
            cur_count['ta'][step] = cur_count['ta'].get(step, [])
            cur_count['ta'][step].append(count)
            cur_start[starting_node] = cur_start.get(starting_node, 0) + 1
            # cur_count['b'][batch_id] = cur_count['b'].get(batch_id, 0) + 1
        if len(walk) > 2:
            t_key = str((walk[-3], walk[-2], walk[-1]))
            self.three_in_row[t_key] = self.three_in_row.get(t_key, 0) + 1
        if len(walk) > 3:
            f_key = str((walk[-4], walk[-3], walk[-2], walk[-1]))
            self.three_in_row[f_key] = self.three_in_row.get(f_key, 0) + 1

    def simulate_walks(self, num_walks, walk_length, concurrent_nodes=16, reuse_probability=0.6):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        random.shuffle(nodes)
        batch = 0
        for node_chunk in self.neighbor_chunker(concurrent_nodes):
            # if batch % 1000 == 0:
            #     print(f"total nodes: {len(nodes)}, processed: {concurrent_nodes * batch}")
            start_nodes = list(node_chunk)
            walks += self.node2vec_walk(walk_length=walk_length,
                                        start_nodes=start_nodes,
                                        num_walks=num_walks,
                                        batch_id=batch,
                                        reuse_probability=reuse_probability)
            batch += 1

        if self.log_stats:
            for v1 in self.edges_count.values():
                for step in set(v1['ta'].keys()):
                    v1['ta'][step] = round(sum(v1['ta'][step]) / len(v1['ta'][step]), 1)
            stats = json.dumps(
                {"nodes": len(G.nodes()),
                 "edges": len(G.edges()),
                 "stats": self.edges_count})
            stats_nodes = json.dumps(self.start_nodes)
            with open(f"bias_edge_{num_walks}_{walk_length}_{reuse_probability}.json", 'w') as stat_file:
                stat_file.write(stats)
            with open(f"bias_nodes_{num_walks}_{walk_length}_{reuse_probability}.json", 'w') as stat_file2:
                stat_file2.write(stats_nodes)
            with open(f"three_count_{num_walks}_{walk_length}_{reuse_probability}.json", 'w') as stat_file3:
                stat_file3.write(json.dumps(self.three_in_row))
            with open(f"four_count_{num_walks}_{walk_length}_{reuse_probability}.json", 'w') as stat_file4:
                stat_file4.write(json.dumps(self.four_in_row))
        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for i, node in enumerate(G.nodes()):
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.random.rand() * K)
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
