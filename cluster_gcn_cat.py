import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time

import dgl
import dgl.nn as dglnn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from ogb.nodeproppred import DglNodePropPredDataset


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(2*in_feats, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(0.5)
        #self.p_list = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        #self.prompt = nn.Linear(2 * n_hidden, n_classes)


    def forward(self, sg, x):

        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)

        return h


dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset("ogbn-products","dataset1"))
graph = dataset[
    0
]  # already prepares ndata['label'/'train_mask'/'val_mask'/'test_mask']



model = SAGE(graph.ndata["feat"].shape[1], 256, dataset.num_classes).cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

num_partitions = 1000
sampler = dgl.dataloading.ClusterGCNSampler(
    graph,
    num_partitions,
    prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
)
# DataLoader for generic dataloading with a graph, a set of indices (any indices, like
# partition IDs here), and a graph sampler.
dataloader = dgl.dataloading.DataLoader(
    graph,
    torch.arange(num_partitions).to("cuda"),
    sampler,
    device="cuda",
    batch_size=100,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    use_uva=True,
)

all_tensors = []

# 循环迭代子图
for sg in dataloader:
    # 获取子图中节点的特征 feat 的形状
    num_nodes = sg.ndata["feat"].shape[0]
    print(sg.ndata["feat"].shape)
    # 创建一个全零张量，维度为 [节点数量, feat_dim]
    tensor = torch.nn.Parameter(torch.Tensor(num_nodes, (256-graph.ndata["feat"].shape[1])))
    nn.init.xavier_uniform_(tensor)
    # 将创建的张量添加到列表中
    all_tensors.append(tensor)

# 创建一个空列表，用于存储所有的拼接结果
all_concatenated = []

# 打印拼接后张量的维度

durations = []
best_test_acc = float('-inf')
test_acc_history = []
for epoch in range(30):
    t0 = time.time()
    model.train()
    for it, sg in enumerate(dataloader):
        x = sg.ndata["feat"]
        y = sg.ndata["label"]
        m = sg.ndata["train_mask"].bool()

        y_hat = model(sg, x)
        loss = F.cross_entropy(y_hat[m], y[m])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 20 == 0:
            acc = MF.accuracy(y_hat[m], y[m])
            mem = torch.cuda.max_memory_allocated() / 1000000
            print("Loss", loss.item(), "Acc", acc.item(), "GPU Mem", mem, "MB")
    tt = time.time() - t0
    print("Run time for epoch# %d: %.2fs" % (epoch, tt))
    durations.append(tt)

    model.eval()
    with torch.no_grad():
        val_preds, test_preds = [], []
        val_labels, test_labels = [], []
        for it, sg in enumerate(dataloader):
            x = sg.ndata["feat"]
            y = sg.ndata["label"]
            m_val = sg.ndata["val_mask"].bool()
            m_test = sg.ndata["test_mask"].bool()
            y_hat = model(sg, x)
            val_preds.append(y_hat[m_val])
            val_labels.append(y[m_val])
            test_preds.append(y_hat[m_test])
            test_labels.append(y[m_test])
        val_preds = torch.cat(val_preds, 0)
        val_labels = torch.cat(val_labels, 0)
        test_preds = torch.cat(test_preds, 0)
        test_labels = torch.cat(test_labels, 0)
        val_acc = MF.accuracy(val_preds, val_labels)
        test_acc = MF.accuracy(test_preds, test_labels)
        test_acc_history.append(test_acc.item())
        print("Validation acc:", val_acc.item(), "Test acc:", test_acc.item())
        if test_acc > best_test_acc:
            best_test_acc = test_acc

print(f"Best Test acc: {best_test_acc:.7f}")


