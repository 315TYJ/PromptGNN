import os

from dgl.data import RedditDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import pandas as pd
import dgl
import dgl.nn as dglnn
#import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from ogb.nodeproppred import DglNodePropPredDataset


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

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset("ogbn-products","dataset1"))  #1000   100   256
#dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv","dataset1"))   # 100 1
#dataset =dgl.data.YelpDataset("dataset1")
#dataset = dgl.data.FlickrDataset("dataset1")   #100  20

#dataset = RedditDataset()
#dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset("ogbn-papers100M","dataset1"))  #10000   class:172
graph = dataset[
    0
]  # already prepares ndata['label'/'train_mask'/'val_mask'/'test_mask']
print("class: ", dataset.num_classes)
model = SAGE(graph.ndata["feat"].shape[1], 256, dataset.num_classes).cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 5e-4)
print(graph.ndata["feat"].shape[1])

num_partitions = 1000
sampler = dgl.dataloading.ClusterGCNSampler(
    graph,
    num_partitions,
    cache_path='products.pkl',
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
for epoch in range(100):
    t0 = time.time()
    model.train()
    for it, sg in enumerate(dataloader):
        x = sg.ndata["feat"]
        y = sg.ndata["label"].to(torch.int64)
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
            y = sg.ndata["label"].to(torch.int64)
            m_val = sg.ndata["val_mask"].bool()
            m_test = sg.ndata["test_mask"].bool()
            y_hat = model(sg, x)
            #print("1",y_hat.shape)
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
        f1_val = calculate_f1_score(val_preds, val_labels.to('cuda'),num_classes=dataset.num_classes)
        f1_test = calculate_f1_score(test_preds, test_labels.to('cuda'),num_classes=dataset.num_classes)
        print("Validation acc:", val_acc.item(), "Test acc:", test_acc.item())
        print("Validation f1:", f1_val, "Test f1:", f1_test)
        test_f1_history.append(f1_test)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if f1_test > best_test_f1:
            best_test_f1 = f1_test

print(f"Best Test acc: {best_test_acc:.7f}")
print(f"Best Test f1: {best_test_f1:.7f}")
print("Mean Time:", np.mean(durations[4:]))

"""def plot_embeddings(model, graph, labels):
    model.eval()
    with torch.no_grad():
        graph = graph.to('cuda')  # Move graph to GPU
        all_feats = graph.ndata["feat"].to('cuda')  # Move features to GPU
        all_labels = labels.to('cuda')  # Move labels to GPU
        embeddings = model(graph, all_feats)

        features_np = embeddings.cpu().numpy()
        features_df = pd.DataFrame(features_np)
        csv_file_path = "node_features.csv"
        features_df.to_csv(csv_file_path, index=False)
plot_embeddings(model, graph, graph.ndata['label'])"""
# 绘制节点特征嵌入散点图
"""def plot_embeddings(model, graph, labels):
    model.eval()
    with torch.no_grad():
        graph = graph.to('cuda')  # Move graph to GPU
        all_feats = graph.ndata["feat"].to('cuda')  # Move features to GPU
        all_labels = labels.to('cuda')  # Move labels to GPU
        m_test = graph.ndata["test_mask"].bool()
        embeddings = model(graph, all_feats)
        embeddings = embeddings[m_test]
        all_labels = all_labels[m_test]
        print("Embeddings shape:", embeddings.shape)  # Debugging line
        embeddings = embeddings.cpu().numpy()  # Move embeddings back to CPU
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        print("t-SNE completed")  # Debugging line

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels.cpu().numpy(), cmap='viridis',
                              alpha=0.7)  # Move labels back to CPU for plotting
        plt.colorbar(scatter, label='Node Labels')
        plt.title('Node Feature Embeddings Scatter Plot')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        # Save the plot as a PDF
        plt.savefig("flickr_muladd.pdf", format='pdf')
        plt.show()


# Assuming you have already defined `model`, `graph`, and `graph.ndata['label']`
plot_embeddings(model, graph, graph.ndata['label'])"""