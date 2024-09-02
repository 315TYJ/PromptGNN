from ogb.nodeproppred import DglNodePropPredDataset
####全图训练没问题，gpu
from dgl.data import FlickrDataset, RedditDataset
import torch
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import torchmetrics.functional as MF
# 构建一个3层的GNN模型
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import time
import utils
class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        """self.layers.append(dglnn.SAGEConv(2*in_feats, n_hidden, "gcn"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "gcn"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "gcn"))"""
        """self.layers.append(dglnn.GraphConv(2*in_feats, n_hidden))
        self.layers.append(dglnn.GraphConv(n_hidden, n_hidden))
        #self.layers.append(dglnn.GraphConv(n_hidden, n_hidden))
        self.layers.append(dglnn.GraphConv(n_hidden, n_hidden))"""
        """self.layers.append(dglnn.GATConv(2*in_feats, n_hidden, 1))
        self.layers.append(dglnn.GATConv(n_hidden, n_hidden, 1))
        # self.layers.append(dglnn.GATConv(n_hidden, n_hidden, 1))
        self.layers.append(dglnn.GATConv(n_hidden, n_hidden, 1))"""
        self.conv1 = dglnn.GCN2Conv(n_hidden, layer=1)
        self.conv2 = dglnn.GCN2Conv(n_hidden, layer=2)
        self.conv3 = dglnn.GCN2Conv(n_hidden, layer=3)
        self.linear1 = nn.Linear(2 * in_feats, n_hidden, bias=False)
        self.linear2 = nn.Linear(n_hidden, n_classes, bias=False)

        self.dropout = nn.Dropout(0.5)
        self.prompt1 = nn.Linear((n_hidden + in_feats), n_classes, bias=False)
        # self.prompt2 = nn.Linear(n_hidden, n_classes, bias=False)
        self.pp = torch.nn.Parameter(torch.Tensor(n_classes, in_feats))
        torch.nn.init.xavier_uniform_(self.pp)
        # for layer in self.layers:
        #     init.xavier_uniform_(layer.weight)
          
    # def reset_parameters(self):
    #     for layer in self.layers:
    #         if isinstance(layer, nn.Linear):
    #             init.xavier_uniform_(layer.weight)  

    def get_prompt(self):

        return self.pp


    def forward(self, sg, x):
        similarity_matrix = torch.mm(x, self.pp.t())  # [B, 47]

        max_similarity_index = torch.argmax(similarity_matrix, dim=1)  # [B]

        selected_class_features = self.pp[max_similarity_index]  # [B, 100]

        x = torch.cat((x, selected_class_features), dim=1)

        x = self.linear1(x)
        h = x
        h = self.conv1(sg, h, x)
        h = self.conv2(sg, h, x)
        h = self.conv3(sg, h, x)
        # h = self.conv4(sg, h, x)
        # h = self.conv5(sg, h, x)
        h = torch.cat((h, selected_class_features), dim=1)
        h = F.relu(h)
        h = self.prompt1(h)

        """similarity_matrix = torch.mm(x, self.pp.t())  # [B, 47]

        max_similarity_index = torch.argmax(similarity_matrix, dim=1)  # [B]

        selected_class_features = self.pp[max_similarity_index]  # [B, 100]

        h = torch.cat((x, selected_class_features), dim=1)
        # h = x + selected_class_features
        # h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        h = F.relu(h)

        #h = torch.cat((h, selected_class_features), dim=1)
        #h = F.relu(h)
        #h = self.prompt1(h)


        # 使用 squeeze 去除大小为 1 的维度
        h = h.squeeze()
        h = h.view(h.size(0), -1)
        #print(h.shape)
        #print(selected_class_features.shape)
        h = torch.cat((h, selected_class_features), dim=1)
        h = F.relu(h)
        h = self.prompt1(h)
        h = h.unsqueeze(1).unsqueeze(2).unsqueeze(3)"""
        return h

    def inference(self, sg, x):
        similarity_matrix = torch.mm(x, self.pp.t())  # [B, 47]

        max_similarity_index = torch.argmax(similarity_matrix, dim=1)  # [B]

        selected_class_features = self.pp[max_similarity_index]  # [B, 100]

        x = torch.cat((x, selected_class_features), dim=1)

        x = self.linear1(x)
        h = x
        h = self.conv1(sg, h, x)
        h = self.conv2(sg, h, x)
        h = self.conv3(sg, h, x)
        # h = self.conv4(sg, h, x)
        # h = self.conv5(sg, h, x)
        h = torch.cat((h, selected_class_features), dim=1)
        h = F.relu(h)
        h = self.prompt1(h)



        """similarity_matrix = torch.mm(x, self.pp.t())  # [B, 47]

        max_similarity_index = torch.argmax(similarity_matrix, dim=1)  # [B]

        selected_class_features = self.pp[max_similarity_index]  # [B, 100]

        h = torch.cat((x, selected_class_features), dim=1)
        # h = x + selected_class_features
        # h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        h = F.relu(h)

        #h = torch.cat((h, selected_class_features), dim=1)
        #h = F.relu(h)
        #h = self.prompt1(h)


        # 使用 squeeze 去除大小为 1 的维度
        h = h.squeeze()
        h = h.view(h.size(0), -1)
        #print(h.shape)
        #print(selected_class_features.shape)
        h = torch.cat((h, selected_class_features), dim=1)
        h = F.relu(h)
        h = self.prompt1(h)
        h = h.unsqueeze(1).unsqueeze(2).unsqueeze(3)"""
        return h

def calculate_f1_score(logits, labels, num_classes):
    preds = torch.argmax(logits, dim=-1)  # 在最后一个维度上取最大值
    preds = preds.view(-1)
    true_positives = torch.zeros(num_classes).to('cuda')
    false_positives = torch.zeros(num_classes).to('cuda')
    false_negatives = torch.zeros(num_classes).to('cuda')
    #print(preds.shape, labels.shape)

    for i in range(num_classes):
        true_positives[i] = torch.sum((preds == i) & (labels == i))
        false_positives[i] = torch.sum((preds == i) & (labels != i))
        false_negatives[i] = torch.sum((preds != i) & (labels == i))

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)

    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    return torch.mean(f1_score).item()

lr= 0.001###0.001
wd= 5e-4
dropout=0.5

dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset("ogbn-products","dataset1"))
#dataset = RedditDataset()
#dataset = dgl.data.FlickrDataset("dataset1")
graph = dataset[0]

#graph = dgl.remove_self_loop(graph)
graph = dgl.add_self_loop(graph)
num_edges = graph.num_edges()
num_nodes = graph.num_nodes()
print("num_nodes", num_nodes)
print("num_edges", num_edges)

# get node feature
feat = graph.ndata['feat'].to('cuda')
# get node labels
labels = graph.ndata['label'].to('cuda')
# get data split
train_mask = graph.ndata['train_mask'].bool()
train_mask = train_mask.to('cuda')
val_mask = graph.ndata['val_mask'].bool()
val_mask = val_mask.to('cuda')
test_mask = graph.ndata['test_mask'].bool()
test_mask = test_mask.to('cuda')
n_features = feat.shape[1]
n_labels = int(labels.max().item() + 1)
graph=graph.to('cuda')
activation=F.relu
model = SAGE(in_feats=n_features, n_hidden=256, n_classes = n_labels)
model = model.to('cuda')
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

Loss = []
Acc_val = []
Acc_test = []
F1_val = []
F1_test = []
durations = []
best_test_acc = float('-inf')
test_acc_history = []
best_test_f1 = float('-inf')
test_f1_history = []
for epoch in range(1000):
    t0 = time.time()
    model.train()
    # 使用所有节点(全图)进行前向传播计算
    logits = model(graph, feat)
    logits = logits.to('cuda')
    logits = logits.view(-1, logits.size(-1))
    # 计算损失值
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])  + 0.01 * utils.constraint(model.get_prompt())
    ##loss = torch.nn.MultiLabelMarginLoss()(logits[train_mask], labels[train_mask])
    ###
    loss = loss.to('cuda')

    # 计算验证集的准确度
    
    # acc_val = evaluate(model, graph, feat, labels, val_mask)
    # acc_test = evaluate(model, graph, feat, labels, test_mask)
    # 进行反向传播计算
    opt.zero_grad()
    loss.backward()
    opt.step()
    mem = torch.cuda.max_memory_allocated() / 1000000
    print(loss.item())
    Loss.append(loss.item())
    model.eval()
    with torch.no_grad():
        logits = model.inference(graph, feat)
        # logits = logits[mask]
        # labels = labels[mask]
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        # return correct.item() * 1.0 / len(labels)
        # acc = MF.accuracy(logits,labels, task="multilabel", num_labels=dataset.num_classes)
        acc_val = MF.accuracy(logits[val_mask], labels[val_mask],   num_classes= n_labels)
        acc_test = MF.accuracy(logits[test_mask], labels[test_mask],   num_classes= n_labels)
        f1_val = calculate_f1_score(logits[val_mask], labels[val_mask].to('cuda'), num_classes=n_labels)
        f1_test = calculate_f1_score(logits[test_mask], labels[test_mask].to('cuda'), num_classes=n_labels)

        print("Test acc:", acc_test.item())
        print("Test F1:", f1_test)

        Acc_val.append(acc_val.item())
        Acc_test.append(acc_test.item())
        if acc_test > best_test_acc:
            best_test_acc = acc_test

        F1_val.append(f1_val)
        F1_test.append(f1_test)
        if f1_test > best_test_f1:
            best_test_f1 = f1_test
        # print(epoch)
    tt = time.time() - t0
    print("Run time for epoch# %d: %.2fs" % (epoch, tt))
    durations.append(tt)
    print("mem", mem)

# print("train",Loss)
# print("val",Acc_val)
# print("test",Acc_test)
print(f"Best Test acc: {best_test_acc:.7f}")
print(f"Best Test f1: {best_test_f1:.7f}")

