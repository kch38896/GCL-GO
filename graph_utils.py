import torch
import numpy as np
import scipy.sparse as sp


def one_hot_encode(go_id):
    labels_onehot = np.identity(n=len(go_id), dtype=np.float)
    one_hot_vector = torch.from_numpy(labels_onehot)
    node_features = one_hot_vector.type(torch.FloatTensor)
    return node_features


def build_adj(idx, edge, go_id):
    adj_new = np.zeros([len(go_id), len(go_id)])

    for i in range(len(idx)):
        adj_new[idx[i][0]][idx[i][1]] = edge[i]
        adj_new[idx[i][1]][idx[i][0]] = edge[i]
    adj_new = np.array(adj_new)
    return adj_new


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(graph_dir, go_id, device):
    idx = []
    edge = []
    with open(graph_dir, "r") as read_in:
        for line in read_in:
            splitted_line = line.split('\t')
            id1 = go_id.index(splitted_line[0].strip())
            id2 = go_id.index(splitted_line[1].strip())
            idx.append([id1, id2])
            edge.append(float(splitted_line[2].strip()))
    # node_features = one_hot_encode(go_id).float()
    adj = build_adj(idx, edge, go_id)
    adj = sp.coo_matrix(adj)
    adj = adj + np.multiply(adj.T, adj.T > adj) - np.multiply(adj, (adj.T > adj))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj).float().to(device)
    node_feat = one_hot_encode(go_id)

    return adj, node_feat


def make_neighbor_graph(h_semantic, go_id, sp_list, device):
    h_semantic_p = []
    for x in go_id:
        h_semantic_p.append((h_semantic[go_id.index(sp_list[x][0])] + h_semantic[go_id.index(sp_list[x][1])] +
                             h_semantic[go_id.index(sp_list[x][2])] + h_semantic[go_id.index(sp_list[x][3])] +
                             h_semantic[go_id.index(sp_list[x][4])]) / 5)
    h_semantic_p = torch.stack(h_semantic_p).to(device)
    # print(h_semantic_p.size()) [num_go_term,128]
    return h_semantic_p
