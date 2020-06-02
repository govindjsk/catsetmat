from collections import defaultdict
from itertools import product
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle

from sklearn.metrics import roc_auc_score
from tqdm.autonotebook import tqdm

vertex_lp_scores = {}

def get_auc_scores(df):
    algos = list(df.columns)
    algos.remove('label')
    auc_scores = {}
    for a in algos:
        auc_scores[a] = roc_auc_score(df['label'], df[a])
    return auc_scores

def get_lp_scores(v, v_, nbrs, nbrs_, mode='hyperedge'):
    if mode == 'vertex':
        if (v, v_) in vertex_lp_scores:
            return vertex_lp_scores[(v, v_)]
    nbrs_v = set(nbrs.get(v, set()))  # Subset of V'
    nbrs_v.discard(v_)

    nbrs_nbrs_v = set()  # Subset of V
    for nv in nbrs_v:  # n_v is an element of V'
        nbrs_nv = set(nbrs_[nv])  # Subset of V
        nbrs_nbrs_v.update(nbrs_nv)
    nbrs_v_ = set(nbrs_.get(v_, set()))  # Subset of V
    nbrs_v_.discard(v)
    nbrs_nbrs_v_ = set()  # Subset of V'
    for nv_ in nbrs_v_:  # n_v_ is an element of V
        nbrs_nv_ = set(nbrs[nv_])  # Subset of V'
        nbrs_nbrs_v_.update(nbrs_nv_)
    # Common Neighbors
    cn = nbrs_nbrs_v.intersection(nbrs_v_)
    cn_ = nbrs_nbrs_v_.intersection(nbrs_v)
    cns, cns_ = len(cn), len(cn_)
    cns_mean = (cns + cns_) / 2

    # Jaccard Coefficient
    union_n = nbrs_nbrs_v.union(nbrs_v_)
    union_n_ = nbrs_nbrs_v_.union(nbrs_v)
    try:
        jcs = cns / len(union_n)
    except ZeroDivisionError:
        jcs = 0.0
    try:
        jcs_ = cns_ / len(union_n_)
    except ZeroDivisionError:
        jcs_ = 0.0
    jcs_mean = (jcs + jcs_) / 2

    # Adamic Adar
    aas = np.sum([1 / (1 + np.log10(1 + len(nbrs.get(w, set())))) for w in cn])
    aas_ = np.sum([1 / (1 + np.log10(1 + len(nbrs_.get(w_, set())))) for w_ in cn_])
    aas_mean = (aas + aas_) / 2
    scores = {'cn': cns, 'cn_': cns_, 'cn_mean': cns_mean,
              'jc': jcs, 'jc_': jcs_, 'jc_mean': jcs_mean,
              'aa': aas, 'aa_': aas_, 'aa_mean': aas_mean}
    if mode == 'vertex':
        vertex_lp_scores[(v, v_)] = scores
    return scores


def incidence_to_hyperedges(S, silent_mode=True, _type=set):
    I, J = S.nonzero()
    hyperedges = defaultdict(set)
    indices = list(zip(I, J))
    if not silent_mode:
        print('Converting incidence matrix to hyperedge {} for faster processing...'.format(_type))
    for i, j in (tqdm(indices) if not silent_mode else indices):
        hyperedges[j].add(i)
    if _type == set:
        return set(map(frozenset, hyperedges.values()))
    elif _type == list:
        return set(map(frozenset, hyperedges.values()))
    elif _type == dict:
        return {i: set(f) for i, f in hyperedges.items()}
    return hyperedges


def get_bipartite_nbrs(B):
    nbrs = incidence_to_hyperedges(B.T, _type=dict)
    nbrs_ = incidence_to_hyperedges(B, _type=dict)
    return nbrs, nbrs_


def data_to_SSB(train_data, test_data, node_list_U, node_list_V):
    n, n_ = len(node_list_U), len(node_list_V)
    train_U, train_V, train_labels = zip(*train_data)
    test_U, test_V, test_labels = zip(*test_data)
    train_pos_hyperedges = []
    test_pos_hyperedges = []
    train_neg_hyperedges = []
    test_neg_hyperedges = []
    for f, f_, label in zip(train_U, train_V, train_labels):
        f = frozenset({x - 1 for x in f.tolist() if x > 0})
        f_ = frozenset({x - 1 for x in f_.tolist() if x > 0})
        if label == 1:
            train_pos_hyperedges.append((f, f_))
        else:
            train_neg_hyperedges.append((f, f_))
    for f, f_, label in zip(test_U, test_V, test_labels):
        f = frozenset({x - 1 for x in f.tolist() if x > 0})
        f_ = frozenset({x - 1 for x in f_.tolist() if x > 0})
        if label == 1:
            test_pos_hyperedges.append((f, f_))
        else:
            test_neg_hyperedges.append((f, f_))
    hyperedges = train_pos_hyperedges + test_pos_hyperedges
    left_hyperedges, right_hyperedges = zip(*hyperedges)
    unique_left_hyperedges = list(set(left_hyperedges))
    left_he_id_map = dict(zip(unique_left_hyperedges, range(len(unique_left_hyperedges))))
    unique_right_hyperedges = list(set(right_hyperedges))
    right_he_id_map = dict(zip(unique_right_hyperedges, range(len(unique_right_hyperedges))))
    m = len(unique_left_hyperedges)
    m_ = len(unique_right_hyperedges)
    S = hyperedges_to_indicence(n, m, unique_left_hyperedges)
    S_ = hyperedges_to_indicence(n_, m_, unique_right_hyperedges)
    train_pos_hyperedge_ids = [(left_he_id_map[f], right_he_id_map[f_]) for f, f_ in train_pos_hyperedges]
    test_pos_hyperedge_ids = [(left_he_id_map[f], right_he_id_map[f_]) for f, f_ in test_pos_hyperedges]
    I, J = zip(*(train_pos_hyperedge_ids+test_pos_hyperedge_ids))
    V = [1]*len(I)
    B = csr_matrix((V, (I, J)), shape=(m, m_))
    train_neg_hyperedge_ids = [(left_he_id_map[f], right_he_id_map[f_]) for f, f_ in train_neg_hyperedges]
    test_neg_hyperedge_ids = [(left_he_id_map[f], right_he_id_map[f_]) for f, f_ in test_neg_hyperedges]
    return S, S_, B, train_pos_hyperedge_ids, train_neg_hyperedge_ids,\
           test_pos_hyperedge_ids, test_neg_hyperedge_ids


def hyperedges_to_indicence(num_nodes, num_hyperedges, hyperedges_list):
    I, J, V = [], [], []
    for i, f in enumerate(hyperedges_list):
        I.extend(list(sorted(f)))
        J.extend([i]*len(f))
        V.extend([1]*len(f))
        S = csr_matrix((V, (I, J)), shape=(num_nodes, num_hyperedges))
    return S


def predict_links(train_data, test_data, U_t, V_t, node_list_U, node_list_V):
    # data_home, data_name, i = [prepared_data_params[x] for x in ['data_home', 'data_name', 'i']]
    # s2slp_data = pickle.load(open(os.path.join(data_home, data_name,
    #                                            '{}.{}.s2slp'.format(data_name, i)), 'rb'))
    S, S_, B, train_pos, train_neg, test_pos, test_neg = data_to_SSB(train_data, test_data, node_list_U, node_list_V)
    train_pairs = train_pos + train_neg
    train_labels = [1] * len(train_pos) + [0] * len(train_neg)
    test_pairs = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)
    A = S * B * S_.T
    B_nbrs, B_nbrs_ = get_bipartite_nbrs(B)
    A_nbrs, A_nbrs_ = get_bipartite_nbrs(A)
    elements = incidence_to_hyperedges(S, _type=dict)
    elements_ = incidence_to_hyperedges(S_, _type=dict)

    def calculate_lp_scores(pairs, labels):
        results = {}
        for (v, v_), l in tqdm(list(zip(pairs, labels))):
            result = {}
            scores = get_lp_scores(v, v_, B_nbrs, B_nbrs_)
            result.update({'B_{}'.format(a): s for a, s in scores.items()})

            B_wo_vv_ = B.copy()
            B_wo_vv_[v, v_] = 0
            A_wo_vv_ = S * B_wo_vv_ * S_.T
            A_wo_vv_nbrs, A_wo_vv_nbrs_ = get_bipartite_nbrs(A_wo_vv_)
            result.update({'A_{}'.format(a): [] for a in scores})
            f_v = set(elements[v])
            f_v_ = set(elements_[v_])  # Set of nodes incident to hyperedge ids v and v_
            count = 0
            for i, j in product(f_v, f_v_):  # i \in f_v, j \in f_v_
                scores = get_lp_scores(i, j, A_nbrs, A_nbrs_)
                _ = {result['A_{}'.format(a)].append(s) for a, s in scores.items()}
                count += 1
            for a in scores:
                result['min_A_{}'.format(a)] = min(result['A_{}'.format(a)])
                result['max_A_{}'.format(a)] = max(result['A_{}'.format(a)])
                result['avg_A_{}'.format(a)] = np.mean(result['A_{}'.format(a)])
                del result['A_{}'.format(a)]
            result.update({'label': l})
            results.update({(v, v_): result})
        df = pd.DataFrame(results).T
        return df

    print('Calculating LP scores for train pairs...')
    train_df = calculate_lp_scores(train_pairs, train_labels)
    print('Calculating LP scores for test pairs...')
    test_df = calculate_lp_scores(test_pairs, test_labels)

    return train_df, test_df
