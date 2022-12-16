import os
import nni
import csv
import time
import json
import argparse
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_RFAGNN import *
from utils import *
from dataset import *

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)
    

def main(param, seed=None):
    # torch.cuda.set_device(3)
    if param['save_mode'] == 0:
        set_seed(param['seed'])
    else:
        set_seed(seed)

    g, features, labels, train_mask, val_mask, test_mask = preprocess_data(param['dataset'], param['train_ratio'], param)

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    g = g.to(device)
    g.ndata['d'] = torch.pow(g.in_degrees().float().clamp(min=1), -0.5)

    model = RFAGNN(g, param).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])

    test_best = 0
    test_val = 0
    min_loss = 100.0
    val_best_epoch = 0

    for epoch in range(param['epochs']):

        model.train()
        optimizer.zero_grad()

        logits = model(features)
        loss_cla = F.nll_loss(logits[train_mask], labels[train_mask])
        loss_graph = model.compute_disentangle_loss()
        train_loss = loss_cla + loss_graph * param['ratio_graph']
        train_acc = accuracy(logits[train_mask], labels[train_mask])

        train_loss.backward()
        optimizer.step()

        model.eval()
        logits = model(features)
        val_loss = F.nll_loss(logits[val_mask], labels[val_mask]).item()
        val_acc = accuracy(logits[val_mask], labels[val_mask])
        test_acc = accuracy(logits[test_mask], labels[test_mask])

        if test_acc > test_best:
            test_best = test_acc

        if val_loss < min_loss:
            min_loss = val_loss
            test_val = test_acc
            val_best_epoch = epoch

        print("\033[0;30;46m Epoch: {} | Loss: {:.6f}, {:.6f}, {:.6f} | Acc: {:.5f}, {:.5f}, {:.5f}, {:.5f}({}), {:.5f} \033[0m".format(
                                    epoch, loss_cla.item(), loss_graph.item(), train_loss.item(), train_acc, val_acc, test_acc, test_val, val_best_epoch, test_best))

    if param['save_mode'] == 0:
        nni.report_final_result(test_val)
        outFile = open('../PerformMetrics.csv','a+', newline='')
        writer = csv.writer(outFile, dialect='excel')
        results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
        for v, k in param.items():
            results.append(k)
        results.append(str(test_acc))
        results.append(str(test_val))
        results.append(str(test_best))
        results.append(str(val_best_epoch))
        writer.writerow(results)
    else:
        return test_acc, test_val, test_best


def evaluate_synthetic(model, data_loder, param):
    with torch.no_grad():

        logits_all = []
        labels_all = []
        model.eval()

        for _, (g, labels, gt_adjs) in enumerate(data_loder):            
            g.ndata['d'] = torch.pow(g.in_degrees().float().clamp(min=1), -0.5)
            model.g = g.to(device)
            
            features = g.ndata['feat'].to(device)
            labels = labels.to(device)
            logits = model(features)
            logits_all.append(logits.detach())
            labels_all.append(labels.detach())
        
        logits_all = torch.cat(tuple(logits_all), 0)
        labels_all = torch.cat(tuple(labels_all), 0)
        micro_f1 = evaluate_f1(logits_all, labels_all)

    return micro_f1


def main_synthetic(param):
    # torch.cuda.set_device(3)
    set_seed(param['seed'])
    log_dir = "../log/{}/".format(param['ExpName'])
    os.makedirs(log_dir, exist_ok=True)

    data = preprocess_data(param['dataset'], param['train_ratio'], param)
    train_data = data[:int(len(data)*0.7)]
    val_data = data[int(len(data)*0.7) : int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]

    train_loader = DataLoader(train_data, batch_size=1000, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=1000, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False, collate_fn=collate)

    model = RFAGNN(None, param).to(device)
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])

    test_best = 0
    test_val = 0
    val_best = 0
    val_best_epoch = 0

    for epoch in range(param['epochs']):

        loss_cla_list = []
        loss_graph_list = []
        train_loss_list = []

        for _, (g, labels, gt_adjs) in enumerate(train_loader):

            model.train()      
            optimizer.zero_grad()
                
            model.g = g.to(device)
            features = g.ndata['feat'].to(device)
            labels = labels.to(device)
            logits = model(features)
            loss_cla = loss_fcn(logits, labels)
            
            loss_graph = model.compute_disentangle_loss()
            train_loss = loss_cla + loss_graph * param['ratio_graph']

            train_loss.backward()
            optimizer.step()
                
            loss_cla_list.append(loss_cla.item())
            loss_graph_list.append(loss_graph.item() * param['ratio_graph'])
            train_loss_list.append(train_loss.item())
            # print("Finished!")

        train_acc = evaluate_synthetic(model, train_loader, param)
        val_acc = evaluate_synthetic(model, val_loader, param)
        test_acc = evaluate_synthetic(model, test_loader, param)

        if test_acc > test_best:
            test_best = test_acc

        if val_acc > val_best:
            val_best = val_acc
            test_val = test_acc
            val_best_epoch = epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'param': param}, log_dir + 'best_model.pt')

        print("\033[0;30;46m Epoch: {} | Loss: {:.6f}, {:.6f}, {:.6f} | Acc: {:.5f}, {:.5f}, {:.5f}, {:.5f}({}), {:.5f} \033[0m".format(
                                    epoch, np.mean(loss_cla_list), np.mean(loss_graph_list), np.mean(train_loss_list), train_acc, val_acc, test_acc, test_val, val_best_epoch, test_best))

    nni.report_final_result(test_val)
    outFile = open('../log/PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in param.items():
        results.append(k)
    results.append(str(test_acc))
    results.append(str(test_val))
    results.append(str(test_best))
    results.append(str(val_best_epoch))
    
    path = '../log/{}/best_model.pt'.format(param['ExpName'])
    best_model = torch.load(path)
    cscore, ged_m, ged_s = evaluate_graph(best_model)
    results.append(str(cscore))
    results.append(str(ged_m))
    results.append(str(ged_s))

    writer.writerow(results)


def evaluate_zinc(model, data_loader):
    loss_fcn = torch.nn.L1Loss()

    model.eval()
    loss = 0
    mae = 0

    with torch.no_grad():
        for batch_idx, (batch_graphs, batch_targets, batch_snorm_n) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            
            model.g = batch_graphs.to(device)
            batch_scores = model.forward(batch_x, batch_snorm_n)
            
            eval_loss = loss_fcn(batch_scores, batch_targets).item()
            eval_mae = F.l1_loss(batch_scores, batch_targets).item()
            loss += eval_loss
            mae += eval_mae
        
    loss /= (batch_idx + 1)
    mae /= (batch_idx + 1)

    return loss, mae


def main_zinc(param):
    # torch.cuda.set_device(3)
    set_seed(param['seed'])
    log_dir = "../log/{}/".format(param['ExpName'])
    os.makedirs(log_dir, exist_ok=True)

    zinc_data = preprocess_data(param['dataset'], param['train_ratio'], param)
    train_loader = DataLoader(zinc_data.train, batch_size=1000, shuffle=True, collate_fn=zinc_data.collate)
    val_loader = DataLoader(zinc_data.val, batch_size=1000, shuffle=False, collate_fn=zinc_data.collate)
    test_loader = DataLoader(zinc_data.test, batch_size=1000, shuffle=False, collate_fn=zinc_data.collate)

    model = RFAGNN(None, param).to(device)
    loss_fcn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)

    test_best = 1e6
    test_val = 1e6
    val_best = 1e6
    val_best_epoch = 0

    for epoch in range(param['epochs']):

        model.train()
        loss_mae_list = []
        loss_graph_list = []
        train_loss_list = []

        for batch_idx, (batch_graphs, batch_targets, batch_snorm_n) in enumerate(train_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            batch_snorm_n = batch_snorm_n.to(device)      
            
            optimizer.zero_grad()
            
            model.g = batch_graphs.to(device)
            batch_scores = model(batch_x, batch_snorm_n)
            
            loss_mae = loss_fcn(batch_scores, batch_targets)
            loss_graph = model.compute_disentangle_loss()
            train_loss = loss_mae + loss_graph * param['ratio_graph']

            train_loss.backward()
            optimizer.step()
            
            loss_mae_list.append(loss_mae.item())
            loss_graph_list.append(loss_graph.item() * param['ratio_graph'])
            train_loss_list.append(train_loss.item())
        
        val_loss, val_mae = evaluate_zinc(model, val_loader)
        test_loss, test_mae = evaluate_zinc(model, test_loader)

        if test_mae < test_best:
            test_best = test_mae

        if val_mae < val_best:
            val_best = val_mae
            test_val = test_mae
            val_best_epoch = epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'param': param}, log_dir + 'best_model.pt')

        print("\033[0;30;46m Epoch: {} | Loss: {:.6f}, {:.6f}, {:.6f} | Acc: {:.5f}, {:.5f}, {:.5f}, {:.5f}({}), {:.5f} \033[0m".format(
            epoch, np.mean(loss_mae_list), np.mean(loss_graph_list), np.mean(train_loss_list), np.mean(loss_mae_list), val_mae, test_mae, test_val, val_best_epoch, test_best))

    outFile = open('../log/PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in param.items():
        results.append(k)
    results.append(str(test_mae))
    results.append(str(test_val))
    results.append(str(test_best))
    results.append(str(val_best_epoch))
    
    path = '../log/{}/best_model.pt'.format(param['ExpName'])
    best_model = torch.load(path)
    cscore, ged_m, ged_s = evaluate_graph(best_model)
    results.append(str(cscore))
    results.append(str(ged_m))
    results.append(str(ged_s))

    nni.report_final_result(ged_m)
    writer.writerow(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='syn-cora', choices = ['cora', 'citeseer', 'pubmed', 'texas', 'cornell', 'wisconsin', 'film', 'chameleon', 'squirrel', 'syn-relation', 'zinc', 'syn-cora'])
    parser.add_argument("--dataset_num", type=int, default=-1)
    parser.add_argument("--graph_mode", type=int, default=0)
    parser.add_argument("--save_mode", type=int, default=1)
    parser.add_argument("--model_mode", type=int, default=1)
    parser.add_argument("--in_dim", type=int, default=30, choices = [1433, 3703, 500, 1703, 1703, 1703, 932, 2325, 2089])
    parser.add_argument("--out_dim", type=int, default=6, choices = [7, 6, 3, 5, 5, 5, 5, 5, 5])
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--graph_dim', type=int, default=32)

    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument("--dataset_name", type=str, default='mixhop-n1490-h1.00-c5-r1')
    parser.add_argument('--homo_ratio', type=float, default=1.0)
    parser.add_argument('--train_ratio', type=float, default=0.48)
    parser.add_argument("--ratio_graph", type=float, default=0.1)
    parser.add_argument("--num_graph", type=int, default=4)
    parser.add_argument("--num_hop", type=int, default=6)
    parser.add_argument("--beta", type=float, default=0.2)

    parser.add_argument('--ExpName', default='run0000')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    if args.dataset == 'cora':
        jsontxt = open("../Param/param_cora.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'citeseer':
        jsontxt = open("../Param/param_citeseer.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'pubmed':
        jsontxt = open("../Param/param_pubmed.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'texas':
        jsontxt = open("../Param/param_texas.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'cornell':
        jsontxt = open("../Param/param_cornell.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'wisconsin':
        jsontxt = open("../Param/param_wisconsin.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'film':
        jsontxt = open("../Param/param_film.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'chameleon':
        jsontxt = open("../Param/param_chameleon.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'squirrel':
        jsontxt = open("../Param/param_squirrel.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'zinc':
        jsontxt = open("../Param/param_zinc.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'syn-relation':
        jsontxt = open("../Param/param_syn-relation.json", 'r').read()
        param = json.loads(jsontxt)
    else:
        param = args.__dict__
    
    param.update(nni.get_next_parameter())
    
    if param['dataset_num'] == 0:
        param['dataset'] = 'texas'
        param['in_dim'] = 1703
    if param['dataset_num'] == 1:
        param['dataset'] = 'cornell'
        param['in_dim'] = 1703
    if param['dataset_num'] == 2:
        param['dataset'] = 'wisconsin'
        param['in_dim'] = 1703
    if param['dataset_num'] == 3:
        param['dataset'] = 'film'
        param['in_dim'] = 932
    if param['dataset_num'] == 4:
        param['dataset'] = 'chameleon'
        param['in_dim'] = 2325
    if param['dataset_num'] == 5:
        param['dataset'] = 'squirrel'
        param['in_dim'] = 2089

    param['save_mode'] = 0
    param['seed'] = args.seed

    if param['save_mode'] == 0:
        if param['dataset'] == 'syn-relation':
            main_synthetic(param)
        elif param['dataset'] == 'zinc':
            main_zinc(param)
        else:
            main(param)
    else:
        test_acc_list = []
        test_val_list = []
        test_best_list = []

        if param['dataset'] == 'syn-cora':
            for i in range(3):
                param['dataset_name'] = "mixhop-n1490-h{:.2f}-c5-r{}".format(param['homo_ratio'], i+1)
                test_acc, test_val, test_best = main(param, param['seed'])
                test_acc_list.append(test_acc)
                test_val_list.append(test_val)
                test_best_list.append(test_best)
        else:
            for seed in range(5):
                test_acc, test_val, test_best = main(param, seed)
                test_acc_list.append(test_acc)
                test_val_list.append(test_val)
                test_best_list.append(test_best)

        nni.report_final_result(np.mean(test_val_list))

        outFile = open('../PerformMetrics.csv','a+', newline='')
        writer = csv.writer(outFile, dialect='excel')
        results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
        for v, k in param.items():
            results.append(k)
        
        results.append(str(test_acc_list))
        results.append(str(test_val_list))
        results.append(str(test_best_list))
        results.append(str(np.mean(test_acc_list)))
        results.append(str(np.mean(test_val_list)))
        results.append(str(np.mean(test_best_list)))
        results.append(str(np.std(test_acc_list)))
        results.append(str(np.std(test_val_list)))
        results.append(str(np.std(test_best_list)))
        writer.writerow(results)
