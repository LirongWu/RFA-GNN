import os
import sys
import csv
import pickle
import torch
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

import dgl
from dgl import DGLGraph
from dgl.data import *


def preprocess_data(dataset, train_ratio, param=None):

    if dataset in ['syn-relation']:
        synthetic_graph = load_synthetic(param)
        data = synthetic_graph.generate()  
        return data


    elif dataset in ['zinc']:
        zinc_data = MoleculeDatasetDGL()
        return zinc_data


    elif dataset in ['syn-cora']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("../syn/syn-cora/{}/{}-sample-cora_row-0.25p__0.5p.{}".format(param['dataset_name'], param['dataset_name'], names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("../syn/syn-cora/{}/{}-sample-cora_row-0.25p__0.5p.test.index".format(param['dataset_name'], param['dataset_name']))
        test_idx_range = np.sort(test_idx_reorder)

        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        if len(test_idx_range_full) != len(test_idx_range):
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended
            non_valid_samples = set(test_idx_range_full) - set(test_idx_range)
        else:
            non_valid_samples = set()

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = graphDict2Adj(graph).astype(np.float32)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        non_valid_samples = non_valid_samples.union(set(list(np.where(labels.sum(1) == 0)[0])))

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))

        train_mask = sample_mask(idx_train, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])
        val_mask = np.bitwise_not(np.bitwise_or(train_mask, test_mask))
        wild_mask = np.bitwise_not(train_mask + val_mask + test_mask)

        for n_i in non_valid_samples:
            if train_mask[n_i]:
                warnings.warn("Non valid samples detected in training set")
                train_mask[n_i] = False
            elif test_mask[n_i]:
                warnings.warn("Non valid samples detected in test set")
                test_mask[n_i] = False
            elif val_mask[n_i]:
                warnings.warn("Non valid samples detected in val set")
                val_mask[n_i] = False
            wild_mask[n_i] = False
  
        features = normalize_features(features.todense())
        features = torch.FloatTensor(features)
        labels = torch.argmax(torch.LongTensor(labels), dim=1)
        train_mask = torch.BoolTensor(train_mask)
        val_mask = torch.BoolTensor(val_mask)
        test_mask = torch.BoolTensor(test_mask)

        row, col = np.where(adj.todense() > 0)
        U = row.tolist()
        V = col.tolist()
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)
        
        return g, features, labels, train_mask, val_mask, test_mask


    elif dataset in ['cora', 'citeseer', 'pubmed']:
        if dataset == 'cora':
            data = citation_graph.load_cora()
        if dataset == 'citeseer':
            data = citation_graph.load_citeseer()
        if dataset == 'pubmed':
            data = citation_graph.load_pubmed()

        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)

        g = data.graph
        g.remove_edges_from(nx.selfloop_edges(g))
        g = DGLGraph(g)
        g.add_edges(g.nodes(), g.nodes())

        return g, features, labels, train_mask, val_mask, test_mask


    elif dataset in ['film']:
        graph_adjacency_list_file_path = '../high_freq/{}/out1_graph_edges.txt'.format(dataset)
        graph_node_features_and_labels_file_path = '../high_freq/{}/out1_node_feature_label.txt'.format(dataset)

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint16)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        row, col = np.where(adj.todense() > 0)

        U = row.tolist()
        V = col.tolist()
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)

        features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])], dtype=float)
        labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])], dtype=int)

        n = labels.shape[0]
        idx = [i for i in range(n)]
        r0 = int(n * train_ratio)
        r1 = int(n * 0.48)
        r2 = int(n * 0.8)

        idx_train = np.array(idx[:r0])
        idx_val = np.array(idx[r1:r2])
        idx_test = np.array(idx[r2:])

        features = normalize_features(features)
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        train = torch.LongTensor(idx_train)
        val = torch.LongTensor(idx_val)
        test = torch.LongTensor(idx_test)

        return g, features, labels, train, val, test


    elif dataset in ['cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel']:

        graph_adjacency_list_file_path = '../high_freq/{}/out1_graph_edges.txt'.format(dataset)
        graph_node_features_and_labels_file_path = '../high_freq/{}/out1_node_feature_label.txt'.format(dataset)

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

        features = normalize_features(features)

        g = DGLGraph(adj)
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)

        n = len(labels.tolist())
        idx = [i for i in range(n)]
        r0 = int(n * train_ratio)
        r1 = int(n * 0.48)
        r2 = int(n * 0.8)
        train = np.array(idx[:r0])
        val = np.array(idx[r1:r2])
        test = np.array(idx[r2:])

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        train = torch.LongTensor(train)
        val = torch.LongTensor(val)
        test = torch.LongTensor(test)

        return g, features, labels, train, val, test


    elif dataset in ['new_chameleon', 'new_squirrel']:
        edge = np.loadtxt('../high_freq/{}/edges.txt'.format(dataset), dtype=int)
        labels = np.loadtxt('../high_freq/{}/labels.txt'.format(dataset), dtype=int).tolist()
        features = np.loadtxt('../high_freq/{}/features.txt'.format(dataset), dtype=float)

        U = [e[0] for e in edge]
        V = [e[1] for e in edge]
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)

        n = len(labels)
        idx = [i for i in range(n)]
        r0 = int(n * train_ratio)
        r1 = int(n * 0.48)
        r2 = int(n * 0.8)
        train = np.array(idx[:r0])
        val = np.array(idx[r1:r2])
        test = np.array(idx[r2:])

        features = normalize_features(features)
        features = torch.FloatTensor(features)

        labels = torch.LongTensor(labels)
        train = torch.LongTensor(train)
        val = torch.LongTensor(val)
        test = torch.LongTensor(test)

        return g, features, labels, train, val, test


class load_synthetic:
    def __init__(self, param):
        self.num_graph = param['num_graph']
        self.seed = param['seed']
        self.saved_file = f'../syn/syn-relation/synthetic_graph_{self.num_graph}_{self.seed}.pkl'
        os.makedirs(os.path.dirname(self.saved_file), exist_ok=True)
        
    def generate(self, graph_size=15, num_graph=10000):

        if os.path.isfile(self.saved_file):
            print(f"load synthetic graph from {self.saved_file}")
            with open(self.saved_file, 'rb') as f:
                return pickle.load(f)
        
        graph_list = load_synthetic.get_graph_list(self.num_graph)
        samples = []

        for _ in range(num_graph):
            union_graph = np.zeros((graph_size, graph_size))
            labels = np.zeros((1, len(graph_list)))
            features = np.random.normal(size=(graph_size, 30), scale=5.0, loc=0)

            factor_graphs = []
            idx_list = list(range(len(graph_list)))
            random.shuffle(idx_list)
            
            for i in range((len(idx_list) + 1) // 2):
                idx = idx_list[i]
                labels[0, idx] = 1

                single_graph = graph_list[idx]
                padded_graph = np.zeros((graph_size, graph_size))
                padded_graph[:single_graph.shape[0], :single_graph.shape[0]] = single_graph
                
                random_index = np.arange(padded_graph.shape[0])
                np.random.shuffle(random_index)
                padded_graph = padded_graph[random_index]
                padded_graph = padded_graph[:, random_index]

                padded_feature = np.random.normal(size=(graph_size, 5), scale=5.0, loc=0)
                padded_feature[:single_graph.shape[0], :] = np.random.normal(size=(single_graph.shape[0], 5), scale=5.0, loc=idx+1)
                features[:, idx*5:(idx+1)*5] = padded_feature[random_index]

                union_graph += padded_graph
                factor_graphs.append((padded_graph, idx))

            g = dgl.DGLGraph()
            g.from_networkx(nx.DiGraph(union_graph))
            g = dgl.transform.add_self_loop(g)
            g.ndata['feat'] = torch.tensor(features).float()
            labels = torch.tensor(labels)
            samples.append((g, labels, factor_graphs))

        with open(self.saved_file, 'wb') as f:
            pickle.dump(samples, f)
            print(f"dataset saved to {self.saved_file}")
            
        return samples


    @staticmethod
    def get_graph_list(num_graph):
        graph_list = []

        g = nx.turan_graph(n=5, r=2)
        graph_list.append(nx.to_numpy_array(g))

        g = nx.house_x_graph()
        graph_list.append(nx.to_numpy_array(g))
        
        g = nx.balanced_tree(r=3, h=2)
        graph_list.append(nx.to_numpy_array(g))
        
        g = nx.grid_2d_graph(m=3, n=3)
        graph_list.append(nx.to_numpy_array(g))

        g = nx.hypercube_graph(n=3)
        graph_list.append(nx.to_numpy_array(g))

        g = nx.octahedral_graph()
        graph_list.append(nx.to_numpy_array(g))

        return graph_list[:num_graph]


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        self.file_path = data_dir + f'/graph_list_labels_{self.split}.pt'

        if not os.path.isfile(self.file_path):
            with open(data_dir + "/%s.pickle" % self.split,"rb") as f:
                self.data = pickle.load(f)

            with open(data_dir + "/%s.index" % self.split,"r") as f:
                data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
                self.data = [ self.data[i] for i in data_idx[0] ]

                assert len(self.data)==num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
        
        self.graph_lists = []
        self.graph_labels = []
        self._prepare()
        self.n_samples = len(self.graph_lists)
        
    def _prepare(self):
        if os.path.isfile(self.file_path):
            print(f"load from {self.file_path}")
            with open(self.file_path, 'rb') as f:
                self.graph_lists, self.graph_labels = pickle.load(f)
            return

        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.data:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])

        with open(self.file_path, 'wb') as f:
            pickle.dump((self.graph_lists, self.graph_labels), f)
            
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]


class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='zinc'):

        data_dir = "../syn/zinc/molecules"
        
        self.train = MoleculeDGL(data_dir, 'train', num_graphs=10000)
        self.val = MoleculeDGL(data_dir, 'val', num_graphs=1000)
        self.test = MoleculeDGL(data_dir, 'test', num_graphs=1000)
        
    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = torch.tensor(np.array(labels)).unsqueeze(1)

        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        tab_sizes_e = [graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        
        return batched_graph, labels, snorm_n


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def graphDict2Adj(graph):
    return nx.adjacency_matrix(nx.from_dict_of_lists(graph), nodelist=range(len(graph)))