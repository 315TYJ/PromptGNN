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
from sklearn.metrics import f1_score
class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        """self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "gcn"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "gcn"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "gcn"))"""
        """self.layers.append(dglnn.GraphConv(in_feats, n_hidden))
        self.layers.append(dglnn.GraphConv(n_hidden, n_hidden))
        self.layers.append(dglnn.GraphConv(n_hidden, n_classes))"""
        """self.layers.append(dglnn.GATConv(in_feats, n_hidden, 1))
        self.layers.append(dglnn.GATConv(n_hidden, n_hidden, 1))
        self.layers.append(dglnn.GATConv(n_hidden, n_classes, 1))"""
        self.conv1 = dglnn.GCN2Conv(n_hidden, layer=1)
        self.conv2 = dglnn.GCN2Conv(n_hidden, layer=2)
        self.conv3 = dglnn.GCN2Conv(n_hidden, layer=3)
        self.linear1 = nn.Linear(in_feats, n_hidden, bias=False)
        self.linear2 = nn.Linear(n_hidden, n_classes, bias=False)

        """self.layers.append(dglnn.GATConv(in_feats, n_hidden, 1))
        self.layers.append(dglnn.GATConv(n_hidden, n_hidden, 1))
        self.layers.append(dglnn.GATConv(n_hidden, n_classes, 1))"""

        self.dropout = nn.Dropout(0.5)
        # for layer in self.layers:
        #     init.xavier_uniform_(layer.weight)
          
    # def reset_parameters(self):
    #     for layer in self.layers:
    #         if isinstance(layer, nn.Linear):
    #             init.xavier_uniform_(layer.weight)  

    # sg为子图
    def forward(self, sg, x):
        x = self.linear1(x)
        h = x
        h = self.conv1(sg, h, x)
        h = self.conv2(sg, h, x)
        h = self.conv3(sg, h, x)
        #h = self.conv4(sg, h, x)
        #h = self.conv5(sg, h, x)
        h = self.linear2(h)

        #h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, sg, x):
        x = self.linear1(x)
        h = x
        h = self.conv1(sg, h, x)
        h = self.conv2(sg, h, x)
        h = self.conv3(sg, h, x)
        #h = self.conv4(sg, h, x)
        #h = self.conv5(sg, h, x)
        h = self.linear2(h)


        #h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
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



def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model.inference(graph, features)
        logits = logits[mask]
        labels = labels[mask]

        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        # return correct.item() * 1.0 / len(labels)
        acc = MF.accuracy(labels, logits)

lr= 0.001###0.001
wd= 5e-4
dropout=0.5


#dataset = dgl.data.AmazonCoBuyComputerDataset("dataset1")
dataset = dgl.data.AmazonCoBuyPhotoDataset("dataset1")
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


n_features = feat.shape[1]
n_labels = int(labels.max().item() + 1)
graph=graph.to('cuda')
activation=F.relu
model = SAGE(in_feats=n_features, n_hidden=256, n_classes = n_labels)
model = model.to('cuda')
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

num_samples = len(graph.ndata["label"])
num_train = int(0.6 * num_samples)
num_val = int(0.2 * num_samples)
num_test = num_samples - num_train - num_val

# 创建随机索引
indices = torch.randperm(num_samples)

# 划分数据集
train_indices = indices[:num_train]
val_indices = indices[num_train:num_train + num_val]
test_indices = indices[num_train + num_val:]

# 将索引转换为tensor，并将其发送到GPU上
train_indices = train_indices
val_indices = val_indices
test_indices = test_indices

# 设置train_mask, val_mask, test_mask
graph.ndata["train_mask"] = torch.zeros(num_samples, dtype=torch.bool).to('cuda')
graph.ndata["val_mask"] = torch.zeros(num_samples, dtype=torch.bool).to('cuda')
graph.ndata["test_mask"] = torch.zeros(num_samples, dtype=torch.bool).to('cuda')

graph.ndata["train_mask"][train_indices] = True
graph.ndata["val_mask"][val_indices] = True
graph.ndata["test_mask"][test_indices] = True




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

for epoch in range(4000):
    t0 = time.time()
    model.train()
    # 使用所有节点(全图)进行前向传播计算
    m_train = graph.ndata["train_mask"].bool()
    logits = model(graph, feat)
    logits = logits.to('cuda')
    logits = logits.view(-1, logits.size(-1))
    # 计算损失值
    loss = F.cross_entropy(logits[m_train], labels[m_train])
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
    Loss.append(loss.item())
    print("Loss", loss.item())
    model.eval()
    with torch.no_grad():
        logits = model.inference(graph, feat)
        m_val = graph.ndata["val_mask"].bool()
        m_test = graph.ndata["test_mask"].bool()
        # logits = logits[mask]
        # labels = labels[mask]
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        # return correct.item() * 1.0 / len(labels)
        # acc = MF.accuracy(logits,labels, task="multilabel", num_labels=dataset.num_classes)
        acc_val = MF.accuracy(logits[m_val], labels[m_val], num_classes=n_labels)
        acc_test = MF.accuracy(logits[m_test], labels[m_test], num_classes=n_labels)

        f1_val = calculate_f1_score(logits[m_val], labels[m_val].to('cuda'), num_classes=n_labels)
        f1_test = calculate_f1_score(logits[m_test], labels[m_test].to('cuda'), num_classes=n_labels)

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

