import argparse

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from torch_sparse import SparseTensor
from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
import utils
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import torch.nn as nn
from logger import Logger
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs1 = GCNConv(in_channels, in_channels)

        #self.prompt1 = nn.Linear((hidden_channels + in_channels), out_channels, bias=False)
        #torch.nn.init.xavier_uniform_(nn.Linear.weight)
        self.dropout = dropout
        #self.dropout1 = nn.Dropout(0.5)
        #self.pp = torch.nn.Parameter(torch.Tensor(self.c, in_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        #torch.nn.init.xavier_uniform_(self.pp)
        #self.pp.data = F.normalize(self.pp.data, p=2, dim=1)
    def get_prompt(self):

        return self.pp,self.prompt1
    def get_prompt1(self):

        return self.pp

    """def initialize_pp(self, features,device):
        # 根据 features 的形状动态设置 self.pp 的形状
        self.pp = nn.Parameter(torch.Tensor(features.size())).to(device)

    def weigth_init(self, graph, x, edge_index, index,device):


        x = self.convs1(x, edge_index)
        h = F.relu(x)
        #h = self.dropout1(h)

        num_samples = min(50, h.size(0))
        random_indices = torch.randperm(h.size(0))[:num_samples]
        features = h[random_indices]


        if self.pp is None:
            self.initialize_pp(features, device)
        del features, x, h, num_samples, random_indices
        #print(features.shape)"""


    def forward(self, x, edge_index):

        #print(self.pp)
        """similarity_matrix = torch.mm(x, self.pp.t())  #

        max_similarity_index = torch.argmax(similarity_matrix, dim=1)  #

        selected_class_features = self.pp[max_similarity_index]  #

        x = x + 0.001*selected_class_features"""
        #x = torch.cat((x, selected_class_features), dim=1)

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        #x = torch.cat((x, selected_class_features), dim=1)

        #x = self.prompt1(x)

        #print(x.shape)
        return x

class Prompt(torch.nn.Module):
    def __init__(self, in_channels):
        super(Prompt, self).__init__()

        self.c = 20

        self.pp = torch.nn.Parameter(torch.Tensor(self.c, in_channels))
        #self.a = torch.nn.Linear(in_channels, self.c)

        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.pp)
        #self.a.reset_parameters()

    def add(self, x: torch.Tensor,device):
        x = x.to(device)

        similarity_matrix = torch.mm(x, self.pp.t())  #

        max_similarity_index = torch.argmax(similarity_matrix, dim=1)  #

        p = self.pp[max_similarity_index]

        return x + p



class GCNInference(torch.nn.Module):
    def __init__(self, weights):
        super(GCNInference, self).__init__()
        self.weights = weights

    def forward(self, x, adj):
        """x = torch.tensor(x, dtype=torch.float32).to(device)
        similarity_matrix = torch.mm(x, pp.t())  #

        max_similarity_index = torch.argmax(similarity_matrix, dim=1)  #

        selected_class_features = pp[max_similarity_index]  #
        x = torch.cat((x, selected_class_features), dim=1)
        x = x.cpu().detach().numpy()"""

        for i, (weight, bias) in enumerate(self.weights):
            x = adj @ x @ weight + bias
            x = np.clip(x, 0, None) if i < len(self.weights) - 1 else x
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.prompt2 = torch.nn.Linear(128, in_channels, bias=False)
        self.prompt3 = torch.nn.Linear(2, out_channels, bias=False)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):

        """pp = self.prompt2(pp)

        similarity_matrix = torch.mm(x_i, pp.t())  #

        max_similarity_index = torch.argmax(similarity_matrix, dim=1)  #

        selected_class_features = pp[max_similarity_index]  #

        x_i = x_i + 0.01*selected_class_features
        x_j = x_j + 0.01 * selected_class_features"""

        """pp_mean = torch.mean(pp, dim=1, keepdim=True)

        similarity_matrix = torch.mm(x_i, pp_mean.t())  #

        max_similarity_index = torch.argmax(similarity_matrix, dim=1)  #

        selected_class_features = pp_mean[max_similarity_index]  #

        x_i = x_i + 0.001 * selected_class_features"""

        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        #print(x.shape)
        #print(selected_class_features.shape)
        """pp_mean = torch.mean(pp, dim=1, keepdim=True)
        similarity_matrix = torch.mm(x, pp_mean.t())  #

        max_similarity_index = torch.argmax(similarity_matrix, dim=1)  #

        selected_class_features = pp_mean[max_similarity_index]
        x = torch.cat((x, selected_class_features), dim=1)
        x = self.prompt3(x)"""
        return torch.sigmoid(x)



def train(model, predictor, prompt, loader, optimizer, device):
    model.train()

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        #pp, prompt1 = model.get_prompt()
        x = prompt.add(data.x,device)
        h = model(x, data.edge_index)

        src, dst = data.edge_index
        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, data.x.size(0), src.size(),
                                dtype=torch.long, device=device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        #loss = pos_loss + neg_loss + utils.constraint(model.get_prompt1())
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = src.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, prompt, data, split_edge, evaluator, batch_size, device):
    predictor.eval()
    print('Evaluating on CPU...')

    weights = [(conv.lin.weight.t().cpu().detach().numpy(),
                conv.bias.cpu().detach().numpy()) for conv in model.convs]
    #pp,prompt1 = model.get_prompt()
    model = GCNInference(weights)
    x = prompt.add(data.x,device)
    x = x.cpu().numpy()

    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])
    adj = adj.set_diag()
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    adj = adj.to_scipy(layout='csr')

    h = torch.from_numpy(model(x, adj)).to(device)

    def test_split(split):
        source = split_edge[split]['source_node'].to(device)
        target = split_edge[split]['target_node'].to(device)
        target_neg = split_edge[split]['target_node_neg'].to(device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return train_mrr, valid_mrr, test_mrr


def main():
    parser = argparse.ArgumentParser(
        description='OGBL-Citation2 (Cluster-GCN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_partitions', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=50)
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-citation2')
    split_edge = dataset.get_edge_split()
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    cluster_data = ClusterData(data, num_parts=args.num_partitions,
                               recursive=False, save_dir=dataset.processed_dir)

    loader = ClusterLoader(cluster_data, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx],
        'target_node_neg': split_edge['valid']['target_node_neg'],
    }

    model = GCN(data.x.size(-1), args.hidden_channels, args.hidden_channels,
                args.num_layers, args.dropout).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)
    prompt = Prompt(data.x.size(-1)).to(device)
    evaluator = Evaluator(name='ogbl-citation2')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        prompt.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()) + list(prompt.parameters()),
            lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            t0 = time.time()
            loss = train(model, predictor, prompt, loader, optimizer, device)
            mem = torch.cuda.max_memory_allocated() / 1000000
            tt = time.time() - t0
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, Mem: {mem:.2f}, Time: {tt:.2f}')

            if epoch > 1 and epoch % args.eval_steps == 0:
                t1 = time.time()
                result = test(model, predictor, prompt, data, split_edge, evaluator,
                              batch_size=64 * 1024, device=device)
                logger.add_result(run, result)

                train_mrr, valid_mrr, test_mrr = result
                tt1 = time.time() - t1
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {train_mrr:.4f}, '
                      f'Valid: {valid_mrr:.4f}, '
                      f'Test: {test_mrr:.4f},'
                      f'Time: {tt1:.2f}')

        print('ClusterGCN')
        logger.print_statistics(run)
    print('ClusterGCN')
    logger.print_statistics()


if __name__ == "__main__":
    main()
