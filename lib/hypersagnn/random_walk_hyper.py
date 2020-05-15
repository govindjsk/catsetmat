from concurrent.futures import as_completed, ProcessPoolExecutor
from scipy.sparse import csr_matrix, lil_matrix
from tqdm.autonotebook import tqdm, trange
import time
import numpy as np
import os
from .utils import mkdir_p

# os.environ["OMP_DISPLAY_ENV"] = "FALSE"
# os.environ["OMP_NUM_THREADS"] = "20"
os.environ["KMP_AFFINITY"] = 'none'
# os.environ["KMP_AFFINITY"]="scatter"


# FIXME: may be there is more efficient method

weight_1st = 1.0
weight_degree = -0.5

# print(weight_1st, weight_degree)


def make_sparse_matrix(raw_data, m, n):
    indptr = [len(row) for row in raw_data]
    indptr = np.cumsum([0] + indptr)
    indices = [i for row in raw_data for i in row]
    data = [1] * len(indices)
    return csr_matrix((data, indices, indptr), shape=(m, n), dtype='float32')


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
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

    return (J, q)


def alias_draw(P):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    J, q = P
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


class HyperGraphRandomWalk():
    def __init__(self, p, q, is_weighted=False):
        self.p = p
        self.q = q
        # FIXME: current version is only for unweighted graph
        self.is_weighted = is_weighted

    def build_graph(self, node_list, edge_list):
        # is considered to be range(num_node) FIXME: maybe a dict for nodes
        # will be better
        self.nodes = node_list
        self.edges = edge_list  # the neighbors of hyperedges (without weight)

        # the neighbors of nodes (with weight)
        n_edge = [[] for _ in range(int(np.max(node_list) + 1))]

        self.node_degree = np.zeros((int(np.max(node_list) + 1)))
        self.edge_degree = np.array([len(e) for e in self.edges])
        for i, e in enumerate(edge_list):
            if isinstance(e, tuple):
                e = list(e)
            e.sort()
            ww = 1  # FIXME: unweighted case
            for v in e:
                n_edge[v].append((i, ww))

                self.node_degree[v] += 1

        for v in node_list:
            n_edge_i = sorted(n_edge[v])
            n_edge[v] = np.array(n_edge_i)

        self.n_edge = n_edge
        # adjacent matrices of V x E, E x V, E x E
        if not SILENT:
            print('adj matrix:')
        self.EV = make_sparse_matrix(
            self.edges, len(
                self.edges), int(
                np.max(node_list) + 1))
        self.delta = lil_matrix((self.EV.shape[0], self.EV.shape[0]))
        size = np.array([1 / np.sqrt(len(e)) for e in self.edges])
        self.delta.setdiag(size)

        self.EV_over_delta = self.delta * self.EV

        self.VE = self.EV.T
        self.VE_over_delta = self.EV_over_delta.T

        if not SILENT:
            print("EV size", self.EV.shape)


def get_first_order_part(nodes):
    alias_n2n_1st = {}
    node2ff_1st = {}

    for src in tqdm(nodes, 'Node') if not SILENT else nodes:
        dsts = node_nbr[src]
        ff_1st = np.array(
            (VE_over_delta[src, :] * EV_over_delta[:, dsts]).todense()).reshape((-1))
        node2ff_1st[src] = ff_1st
        unnormalized_probs = ff_1st / np.sqrt(node_degree[dsts])
        normalized_probs = unnormalized_probs / np.sum(unnormalized_probs)
        alias_n2n_1st[src] = alias_setup(normalized_probs)

    return alias_n2n_1st, node2ff_1st


def get_first_order(G):
    if not SILENT:
        print("1st order: ")
    global EV, VE, EV_over_delta, VE_over_delta, node_nbr, node_degree

    EV = G.EV
    VE = G.VE
    EV_over_delta = G.EV_over_delta
    VE_over_delta = G.VE_over_delta
    node_nbr = G.node_nbr
    node_degree = G.node_degree
    nodes = np.copy(G.nodes)
    np.random.shuffle(nodes)
    try:
        processes_num = 80
        pool = ProcessPoolExecutor(max_workers=processes_num)
        split_num = min(processes_num, int(len(nodes) / 100)) + 1
        if not SILENT:
            print("split_num", split_num)
        nodes = np.array_split(nodes, split_num)
        if not SILENT:
            print("Start get first order")
        process_list = []
        for node in nodes:
            process_list.append(pool.submit(get_first_order_part, node))
        alias_n2n_1st = {}
        node2ff_1st = {}
        for p in as_completed(process_list):
            alias_t1, alias_t2 = p.result()
            alias_n2n_1st.update(alias_t1)
            node2ff_1st.update(alias_t2)
        pool.shutdown(wait=True)
    except ValueError:
        alias_n2n_1st, node2ff_1st = get_first_order_part(nodes)
    if not SILENT:
        print("start turn dict to list")
    nodes = np.copy(G.nodes)
    alias_n2n_1st_list = [[] for n in nodes]
    node2ff_1st_list = [[] for n in nodes]
    for n in nodes:
        alias_n2n_1st_list[n] = alias_n2n_1st[n]
        node2ff_1st_list[n] = node2ff_1st[n]
    return alias_n2n_1st_list, node2ff_1st_list


def get_src_dst2e(G, edges):
    src_dst_2e = {}
    node_nbr = [[] for n in range(int(np.max(G.nodes)) + 1)]

    for e1 in tqdm(edges, "Edge") if not SILENT else edges:
        for src in G.edges[e1]:
            for dst in G.edges[e1]:
                if src != dst:
                    if (src, dst) in src_dst_2e:
                        src_dst_2e[(src, dst)].append(e1)
                    else:
                        src_dst_2e[(src, dst)] = [e1]

                    node_nbr[src].append(dst)
                    node_nbr[dst].append(src)

    if not SILENT:
        print("get node nbr")

    for k in trange(len(node_nbr)) if not SILENT else range(len(node_nbr)):
        list1 = node_nbr[k]
        list1 = sorted(set(list1))
        node_nbr[k] = list1
    for k in tqdm(src_dst_2e.keys(), 'src_dst') if not SILENT else src_dst_2e.keys():
        list1 = sorted(src_dst_2e[k])
        src_dst_2e[k] = list1
    G.src_dst_2e = src_dst_2e
    G.node_nbr = np.array(node_nbr)


def get_alias_n2n_2nd(src, dst):
    dst_nbr = node_nbr[dst]

    pp = np.ones(len(dst_nbr))
    pp /= q

    e1_all = src_dst_2e[(src, dst)]
    # ff_all_1 = EV[e1_all, :dst] * VE[:dst]
    # ff_all_2 = EV[e1_all, dst+1:] * VE[dst+1:]
    condition = np.array(VE[dst_nbr, :][:, e1_all].sum(axis=-1)).reshape((-1))
    pp[condition > 0] /= p

    for i, nb in enumerate(dst_nbr):
        if nb == src:
            pp[i] *= q
        elif (src, nb) in src_dst_2e:
            pp[i] *= q
        # e2_all = src_dst_2e[(dst, nb)]
        # ff_all_1 = EV[e1_all, :dst] * VE[:dst, e2_all]
        # ff_all_2 = EV[e1_all, dst+1:] * VE[dst+1:, e2_all]
        #
        #
        # pp[i] *= ((ff_all_1.sum() + ff_all_2.sum()) ** 0.5)

    ff_1st = node2ff_1st[dst]
    # pp += np.random.randn(pp.shape[0]) * 0.5
    pp *= (ff_1st ** weight_1st)
    pp *= (node_degree[dst_nbr] ** weight_degree)

    unnormalized_probs = pp
    normalized_probs = unnormalized_probs / np.sum(unnormalized_probs)
    normalized_probs = normalized_probs / np.sum(normalized_probs)
    return alias_setup(normalized_probs)


def get_alias_n2n_2nd_dropped(src, dst):
    dst_nbr = node_nbr[dst]
    pp = np.zeros(len(dst_nbr))
    e1_all = src_dst_2e[(src, dst)]
    # ff_all_1 = EV[e1_all, :dst] * VE[:dst]
    # ff_all_2 = EV[e1_all, dst+1:] * VE[dst+1:]
    condition = np.array(VE[dst_nbr, :][:, e1_all].sum(axis=-1)).reshape((-1))
    pp[condition > 0] += p * condition[condition > 0]

    for i, nb in enumerate(dst_nbr):
        if nb == src:
            pp[i] += node_degree[src]
        elif (src, nb) in src_dst_2e:
            pp[i] += len(src_dst_2e[(src, nb)])
        else:
            pp[i] += 1 / q
    # e2_all = src_dst_2e[(dst, nb)]
    # ff_all_1 = EV[e1_all, :dst] * VE[:dst, e2_all]
    # ff_all_2 = EV[e1_all, dst+1:] * VE[dst+1:, e2_all]
    #
    #
    # pp[i] *= ((ff_all_1.sum() + ff_all_2.sum()) ** 0.5)

    ff_1st = node2ff_1st[dst]
    # pp += np.random.randn(pp.shape[0]) * 0.5
    pp *= (ff_1st ** weight_1st)
    pp *= (node_degree[dst_nbr] ** weight_degree)

    unnormalized_probs = pp
    normalized_probs = unnormalized_probs / np.sum(unnormalized_probs)
    normalized_probs = normalized_probs / np.sum(normalized_probs)
    return alias_setup(normalized_probs)


def get_second_order(nodes):
    alias_n2n_2nd = {}
    for i in trange(len(nodes)) if not SILENT else range(len(nodes)):
        src = nodes[i]
        dsts = node_nbr[src]

        for dst_index, dst in enumerate(dsts):
            alias_n2n_2nd[(src, dst)] = get_alias_n2n_2nd(src, dst)
    return alias_n2n_2nd
# for multi-processing


def parallel_get_second_order(G):
    if not SILENT:
        print("2nd order: ")
    global p, q, node_nbr, VE, EV, src_dst_2e, node2ff_1st, node_degree, node_nbr
    p, q = G.p, G.q
    node_nbr = G.node_nbr
    VE = G.VE
    EV = G.EV
    src_dst_2e = G.src_dst_2e
    node2ff_1st = G.node2ff_1st
    node_degree = G.node_degree
    node_nbr = G.node_nbr

    # f is a csr-matrix
    # O(\sum_v (\sum_e\in nbr(v) |e|)^2)
    second_start = time.time()
    nodes = np.copy(G.nodes)
    np.random.shuffle(nodes)
    try:
        processes_num = 80
        pool = ProcessPoolExecutor(max_workers=processes_num)

        split_num = min(processes_num, int(len(nodes) / 100)) * 2 + 1
        if not SILENT:
            print("split_num", split_num)
        nodes = np.array_split(nodes, split_num)
        if not SILENT:
            print("Start get second order alias")
        process_list = []
        for node in nodes:
            process_list.append(pool.submit(get_second_order, node))
        alias_n2n_2nd = {}
        for p in as_completed(process_list):
            alias_t1 = p.result()
            alias_n2n_2nd.update(alias_t1)
        pool.shutdown(wait=True)
    except ValueError:
        alias_n2n_2nd = get_second_order(nodes)
    if not SILENT:
        print("get-second-order-term running time: " +
              str(time.time() - second_start))
    if not SILENT:
        print("Start to turn the dict into list")
    alias_n2n_2nd_list = []
    alias_n2n_toid = {}
    for i, k in enumerate(tqdm(alias_n2n_2nd.keys(), 'n2n') if not SILENT else alias_n2n_2nd.keys()):
        alias_n2n_toid[k] = i
        alias_n2n_2nd_list.append(alias_n2n_2nd[k])
    G.alias_n2n_toid = alias_n2n_toid
    G.alias_n2n_2nd_list = alias_n2n_2nd_list
    return alias_n2n_2nd


def random_walk_list(walk_length, start):
    walk = [start]
    while len(walk) < (walk_length):
        cur = walk[-1]
        cur_ns = node_nbr[cur]
        if len(cur_ns) < 1:
            walk.append(cur)
            continue

        try:
            if len(walk) == 1:
                next_n = cur_ns[alias_draw(alias_n2n_1st[cur])]
            else:
                prev_n = walk[-2]
                next_n = cur_ns[alias_draw(
                    alias_n2n_2nd_list[alias_n2n_toid[(prev_n, cur)]])]

        except Exception as e:
            print("error", e)
            break
        walk.append(next_n)

    return walk


def simulate_walks_part(num_walks, walk_length, nodes):
    walks = []
    for node in tqdm(nodes, 'Node') if not SILENT else nodes:
        for walk_iter in range(num_walks):
            walk = random_walk_list(walk_length, node)
            walks.append(walk)
    return walks


def simulate_walks_para(G, num_walks, walk_length):
    """
    Repeatedly simulate random walks from each node.
    """
    global alias_n2n_1st, alias_n2n_2nd_list, alias_n2n_toid
    alias_n2n_1st = G.alias_n2n_1st
    alias_n2n_2nd_list = G.alias_n2n_2nd_list
    alias_n2n_toid = G.alias_n2n_toid
    nodes = np.copy(G.nodes)

    if os.name != 'nt':
        processes_num = 30
        pool = ProcessPoolExecutor(max_workers=processes_num)
        if not SILENT:
            print("sample walks:")
        split_num = processes_num
        if not SILENT:
            print("split_num", split_num)
        np.random.shuffle(nodes)
        nodes = np.array_split(nodes, split_num)
        process_list = []
        for node in nodes:
            process_list.append(
                pool.submit(
                    simulate_walks_part,
                    num_walks,
                    walk_length,
                    node))
        walks = []
        for p in as_completed(process_list):
            alias_t1 = p.result()
            walks += alias_t1
        pool.shutdown(wait=True)
    else:
        walks = simulate_walks_part(num_walks, walk_length, nodes)
    if not SILENT:
        print("start permutation")
    idx = np.random.permutation(len(walks))
    walks = np.array(walks, dtype='int')
    return walks[idx]


def toint(hyperedge_list):
    # print(hyperedge_list)
    return np.array([h.astype('int') for h in hyperedge_list])


def random_walk_hyper(args, node_list, hyperedge_list, data_name, set_name, split_id, base_path, silent=False):
    global SILENT
    SILENT = silent
    p, q = args.p, args.q

    num_walks, walk_length, window_size = args.num_walks, args.walk_length, args.window_size
    walks_save_path = os.path.join(base_path, 'walks/{}/p{}_q{}_r{}_l{}_{}{}_hyper_walks.txt'.format(
        data_name, p, q, num_walks, walk_length, set_name, split_id))
    # if not os.path.exists("walks/{}/".format(args.data)):
    mkdir_p(os.path.join(base_path, "walks/{}/".format(data_name)))
    start = time.time()

    if not args.TRY and os.path.exists(walks_save_path):
        return walks_save_path
    else:
        G = HyperGraphRandomWalk(p, q)
        G.data = data_name
        # FIXME: take care when the input are tensors, but I think other
        # dataset they will not be
        if not silent:
            print('build')
        hyperedge_list = toint(hyperedge_list)
        G.build_graph(node_list, hyperedge_list)
        edges = np.array(range(len(G.edges)))
        if not silent:
            print("Building pairwise to hyper dict")
        get_src_dst2e(G, edges)
        G.alias_n2n_1st, G.node2ff_1st = get_first_order(G)
        parallel_get_second_order(G)
        if not silent:
            print("RandomWalk getting edges time: %.2lf" % (time.time() - start))
            print(G.__dict__.keys())

        name = [
            'data',
            'edges',
            'node_degree',
            'edge_degree',
            'n_edge',
            'EV',
            'delta',
            'EV_over_delta',
            'VE',
            'VE_over_delta',
            'src_dst_2e',
            'node_nbr',
            'node2ff_1st']

        for n in name:
            delattr(G, n)

        walks = simulate_walks_para(G, num_walks, walk_length)
        if not silent:
            print("RandomWalk running time: %.2lf" % (time.time() - start))
        np.savetxt(walks_save_path, walks, fmt="%d", delimiter=" ")
        # np.save(walks_save_path,walks)
        del G
        del walks
        if not silent:
            print("RandomWalk running time: %.2lf" % (time.time() - start))

        return walks_save_path
