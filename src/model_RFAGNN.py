import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import function as fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FALayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_graph, activation, param):
        super(FALayer, self).__init__()
        self.param = param
        self.num_graph = num_graph

        if self.param['dataset'] == 'syn-relation' or self.param['dataset'] == 'zinc':
            self.apply_mod = nn.ModuleList()
            for num in range(self.num_graph):
                self.apply_mod.append(NodeApplyModule(in_dim, out_dim, activation))
            
        if self.param['model_mode'] == 1:
            self.num_index = 0
            self.k_index = 0
            self.gate = nn.ModuleList()
            for num in range(self.num_graph * self.param['num_hop']):
                self.gate.append(nn.Linear(2 * in_dim, 1).to(device))

    def edge_applying(self, edges):
        h = torch.cat([edges.dst[f'feature_{self.num_index}_{self.k_index}'], edges.src[f'feature_{self.num_index}_{self.k_index}']], dim=1)
        if self.param['dataset'] == 'syn-relation' or self.param['dataset'] == 'zinc':
            e = torch.tanh(self.gate[self.k_index+self.param['num_hop']*self.num_index](h)).squeeze()
        else:
            e = torch.tanh(self.gate[self.k_index+self.param['num_hop']*self.num_index](h)).squeeze() * edges.dst['d'] * edges.src['d']
            e = F.dropout(e, p=self.param['dropout'], training=self.training)
        return {f"factor_{self.num_index}_{self.k_index}": e}

    def forward(self, g, x):

        out_features = []
        norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5).view(-1, 1).to(device)

        for num in range(self.num_graph):
            g.ndata.update({f'feature_{num}_0': x})

            for k in range(self.param['num_hop']):
                if self.param['model_mode'] == 1:
                    self.num_index = num
                    self.k_index = k
                    g.apply_edges(self.edge_applying)

                if self.param['dataset'] == 'syn-relation':
                    hidden = g.ndata[f'feature_{num}_{k}']
                    g.ndata[f'feature_{num}_{k+1}'] = hidden * norm
                else:
                    g.ndata[f'feature_{num}_{k+1}'] = g.ndata[f'feature_{num}_{k}']

                if self.param['model_mode'] == 0 or k == 0:
                    g.update_all(fn.u_mul_e(f'feature_{num}_{k+1}', f"factor_{num}", 'm'), fn.sum('m', f'feature_{num}_{k+1}'))
                else:
                    g.update_all(fn.u_mul_e(f'feature_{num}_{k+1}', f"factor_{num}_{k}", 'm'), fn.sum('m', f'feature_{num}_{k+1}'))
                g.ndata[f'feature_{num}_{k+1}'] = g.ndata[f'feature_{num}_{k+1}'] * (1.0 - self.param['beta']) + g.ndata[f'feature_{num}_{0}'] * self.param['beta']

            last_one = self.param['num_hop']
            if self.param['dataset'] == 'syn-relation' or self.param['dataset'] == 'zinc':
                out = self.apply_mod[num](g.ndata[f'feature_{num}_{last_one}'])
            else:
                out = g.ndata[f'feature_{num}_{last_one}']
            out_features.append(out)

        out = torch.cat(tuple([rst for rst in out_features]), -1)

        return out


class RFAGNN(nn.Module):
    def __init__(self, g, param):
        super(RFAGNN, self).__init__()

        self.g = g
        self.param = param
        self.in_dim = param['in_dim']
        self.graph_dim = param['graph_dim']
        self.hidden_dim = param['hidden_dim']
        self.out_dim = param['out_dim']
        
        self.num_graph = param['num_graph']
        self.dropout = param['dropout']
        
        self.GraphLearning = GraphLearning(self.in_dim, self.graph_dim, self.num_graph, param)

        if self.param['dataset'] == 'syn-relation':
            self.layers = nn.ModuleList()
            self.layers.append(FALayer(self.hidden_dim, self.hidden_dim, self.num_graph, nn.LeakyReLU(negative_slope=0.2), param))
            # self.layers.append(FALayer(self.hidden_dim * self.num_graph, self.hidden_dim, self.num_graph, param))
        elif param['dataset'] == 'zinc':
            self.activate = nn.LeakyReLU(negative_slope=0.2)
            self.embedding = nn.Embedding(28, self.in_dim)

            self.layers = nn.ModuleList()
            self.layers.append(FALayer(self.hidden_dim, self.hidden_dim, self.num_graph, None, param))
            self.layers.append(FALayer(self.hidden_dim * self.num_graph, self.hidden_dim, self.num_graph, None, param))
            self.layers.append(FALayer(self.hidden_dim * self.num_graph, self.hidden_dim, self.num_graph, None, param))
            self.layers.append(FALayer(self.hidden_dim * self.num_graph, self.hidden_dim, self.num_graph, None, param))

            self.BNs = nn.ModuleList()
            self.BNs.append(nn.BatchNorm1d(self.hidden_dim * self.num_graph))
            self.BNs.append(nn.BatchNorm1d(self.hidden_dim * self.num_graph))
            self.BNs.append(nn.BatchNorm1d(self.hidden_dim * self.num_graph))
            self.BNs.append(nn.BatchNorm1d(self.hidden_dim * self.num_graph))

            self.regressor1 = nn.Linear(self.hidden_dim * self.num_graph, self.hidden_dim)
            self.regressor2 = nn.Linear(self.hidden_dim, self.out_dim)
        else:
            self.FALayer = FALayer(self.hidden_dim, self.hidden_dim, self.num_graph, nn.LeakyReLU(negative_slope=0.2), param)

        self.linear1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim * self.num_graph, self.out_dim)

    def forward(self, x, snorm_n=None):
        if self.param['dataset'] == 'zinc':
            x = self.embedding(x)
            self.g = self.GraphLearning(self.g, x)

            h1 = F.dropout(x, p=self.dropout, training=self.training)
            h2 = self.linear1(h1)
            
            for layer, bn in zip(self.layers, self.BNs):
                h2 = layer(self.g, h2)
                h2 = h2 * snorm_n
                h2 = self.activate(bn(h2))

            self.g.ndata['h_mean'] = F.dropout(h2, p=self.dropout, training=self.training)
            h3 = dgl.mean_nodes(self.g, 'h_mean')
            h3 = torch.relu(h3)
            h3 = self.regressor1(h3)
            h3 = torch.relu(h3)
            out = self.regressor2(h3)

            return out

        h1 = F.dropout(x, p=self.dropout, training=self.training)
        self.g = self.GraphLearning(self.g, h1)
        
        h2 = F.dropout(x, p=self.dropout, training=self.training)
        h2 = torch.relu(self.linear1(h2))

        if self.param['dataset'] == 'syn-relation':
            for layer in self.layers:
                h2 = F.dropout(h2, p=self.dropout, training=self.training)
                h2 = layer(self.g, h2)

            self.g.ndata['h_mean'] = F.dropout(h2, p=self.dropout, training=self.training)
            h3 = dgl.mean_nodes(self.g, 'h_mean')
            h3 = torch.tanh(h3)
            out = self.linear2(h3)
            return out
        else:
            h2 = F.dropout(h2, p=self.dropout, training=self.training)
            h2 = self.FALayer(self.g, h2)
            h2 = self.linear2(h2)
            out = F.log_softmax(h2, 1)
            return out

    def compute_disentangle_loss(self):
        loss_graph = self.GraphLearning.compute_disentangle_loss(self.g)
        return loss_graph

    def get_factor(self):
        factor_list = [self.g]
        return factor_list


class GraphLearning(nn.Module):
    def __init__(self, in_dim, graph_dim, num_graph, param):
        super(GraphLearning, self).__init__()

        self.num_graph = num_graph
        self.param = param
        self.dropout = param['dropout']

        self.linear = nn.Linear(in_dim, graph_dim).to(device)

        self.gate = nn.ModuleList()
        for num in range(self.num_graph):
            self.gate.append(nn.Linear(2 * graph_dim, 1).to(device))

        self.GraphAE = GraphEncoder(graph_dim, graph_dim // 2, param).to(device)
        if self.param['graph_mode'] == 0:
            self.classifier = nn.Linear(graph_dim, num_graph).to(device)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.classifier = nn.Linear(graph_dim, 1).to(device)
            self.loss_fn = nn.MSELoss()

    def edge_applying(self, edges):
        factor_dict = {}

        h = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        for num in range(self.num_graph):
            if self.param['dataset'] == 'syn-relation' or self.param['dataset'] == 'zinc':       
                e = torch.sigmoid(self.gate[num](h)).squeeze()
            else:
                e = torch.tanh(self.gate[num](h)).squeeze() * edges.dst['d'] * edges.src['d']
                e = F.dropout(e, p=self.dropout, training=self.training)
            factor_dict.update({f"factor_{num}": e})

        return factor_dict

    def forward(self, g, x):

        h = self.linear(x)
        g.ndata['h'] = h
        g.apply_edges(self.edge_applying)
        self.hidden = h
  
        return g

    def compute_disentangle_loss(self, g):
        factors_feature = [self.GraphAE(g, self.hidden, f"factor_{num}") for num in range(self.num_graph)] 
        labels = [torch.ones(f.shape[0])*i for i, f in enumerate(factors_feature)]
        labels = torch.cat(tuple(labels), 0).long().to(device)

        factors_feature = torch.cat(tuple(factors_feature), 0)
        pred = self.classifier(factors_feature)
        if self.param['graph_mode'] == 0:
            pred = nn.Softmax(dim=1)(pred)
            loss_graph = self.loss_fn(pred, labels)
        else:
            loss_graph_list = []
            for i in range(self.num_graph-1):
                for j in range(i+1, self.num_graph):
                    loss_graph_list.append(self.loss_fn(pred[i], pred[j]))
            loss_graph_list = torch.Tensor(loss_graph_list)
            loss_graph = -torch.sum(loss_graph_list)

        return loss_graph


class NodeApplyModule(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim).to(device)
        self.activation = activation

    def forward(self, node_features):
        h = self.linear(node_features)
        if self.activation is not None:
            h = self.activation(h)
        return h


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, param):
        super(GraphEncoder, self).__init__()
        self.param = param
        self.apply_mod1 = NodeApplyModule(input_dim, hidden_dim, F.tanh)
        self.apply_mod2 = NodeApplyModule(hidden_dim, input_dim, F.tanh)

    def forward(self, g, features, factor_key):
        g = g.local_var()

        norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5).view(-1, 1).to(device)

        if self.param['dataset'] == 'syn-relation':
            g.ndata.update({'h': features * norm})
        else:
            g.ndata.update({'h': features})
        g.update_all(fn.u_mul_e('h', factor_key, 'm'), fn.sum('m', 'h'))
        features = self.apply_mod1(g.ndata['h'])

        if self.param['dataset'] == 'syn-relation':
            g.ndata.update({'h': features * norm})
        else:
            g.ndata.update({'h': features})
        g.update_all(fn.u_mul_e('h', factor_key, 'm'), fn.sum('m', 'h'))
        g.ndata['h'] = self.apply_mod2(g.ndata['h'])

        h = dgl.mean_nodes(g, 'h')

        return h