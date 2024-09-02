import os

from dgl.data import RedditDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time

import dgl
import dgl.nn as dglnn
#import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import torch
#import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from ogb.nodeproppred import DglNodePropPredDataset
import utils
import dgl.function as fn
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
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
        self.layers.append(dglnn.SAGEConv(2*in_feats, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.c = n_classes
        #self.c = 10
        self.linear = nn.Linear(in_feats, n_classes, bias=False)
        #self.c = 10
        self.dropout = nn.Dropout(0.5)
        self.prompt1 = nn.Linear((n_hidden + in_feats), n_classes, bias=False)
        #self.prompt2 = nn.Linear(n_hidden, n_classes, bias=False)
        self.pp = torch.nn.Parameter(torch.Tensor(self.c, in_feats))

    def weigth_init(self, graph, inputs, label, index):
        graph = graph.to(torch.device('cuda:0'))
        h = inputs.to(torch.device('cuda:0'))
        h = self.dropout(h)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        h = F.relu(h)
        graph.ndata['h'] = h
        graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neighbor'))
        neighbor = graph.ndata['neighbor']
        h = torch.cat((h, neighbor), dim=1)

        features = h[index]
        labels = label[index.long()]
        p = []
        for i in range(self.n_classes):
            p.append(features[labels == i].mean(dim=0).view(1, -1))
        temp = torch.cat(p, dim=0)

        self.pp.data = temp.to(torch.device('cuda:0'))

    def get_prompt(self):

        return self.pp

    def forward(self, sg, x):

        similarity_matrix = torch.mm(x, self.pp.t())  # [B, 47]

        max_similarity_index = torch.argmax(similarity_matrix, dim=1)  # [B]

        selected_class_features = self.pp[max_similarity_index]  # [B, 100]

        h = torch.cat((x, selected_class_features), dim=1)
        # h = x + selected_class_features
        # h = x * selected_class_features
        alpha = 0.8  # 权重因子
        # h = alpha * x + (1 - alpha) * selected_class_features
        # h = x + selected_class_features
        # h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        h = F.relu(h)
        h = torch.cat((h, selected_class_features), dim=1)
        h = F.relu(h)
        h = self.prompt1(h)

        return h




dataset = dgl.data.AmazonCoBuyComputerDataset("dataset1")
#dataset = dgl.data.AmazonCoBuyPhotoDataset("dataset1")

"""data = DglNodePropPredDataset("ogbn-products", 'dataset1')
splitted_idx = data.get_idx_split()
train_idx, val_idx, test_idx = (
    splitted_idx["train"],
    splitted_idx["valid"],
    splitted_idx["test"],
)
_, labels = data[0]
labels = labels[:, 0]"""

graph = dataset[
    0
]  # already prepares ndata['label'/'train_mask'/'val_mask'/'test_mask']
n_hidden = 256
model = SAGE(graph.ndata["feat"].shape[1], n_hidden, dataset.num_classes).cuda()
#model.weigth_init(graph, graph.ndata["feat"], labels, train_idx)
#prompt = prompt(graph.ndata["feat"].shape[1], 10, dataset.num_classes).cuda()
#prompt.weigth_init(graph, graph.ndata["feat"], labels, train_idx)
# 将prompt添加到模型参数列表中
#parameters = list(model.parameters()) + list(prompt.parameters())
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 5e-4)
print(graph.ndata["feat"].shape[1])

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
graph.ndata["train_mask"] = torch.zeros(num_samples, dtype=torch.bool)
graph.ndata["val_mask"] = torch.zeros(num_samples, dtype=torch.bool)
graph.ndata["test_mask"] = torch.zeros(num_samples, dtype=torch.bool)

graph.ndata["train_mask"][train_indices] = True
graph.ndata["val_mask"][val_indices] = True
graph.ndata["test_mask"][test_indices] = True

num_partitions = 100
sampler = dgl.dataloading.ClusterGCNSampler(
    graph,
    num_partitions,
    cache_path='amazon_co_buy_computer.pkl',
    prefetch_ndata=["feat", "label"],
)
# DataLoader for generic dataloading with a graph, a set of indices (any indices, like
# partition IDs here), and a graph sampler.
dataloader = dgl.dataloading.DataLoader(
    graph,
    torch.arange(num_partitions).to("cuda"),
    sampler,
    device="cuda",
    batch_size=20,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    use_uva=True,
)



Acc_val = []
Acc_test = []
F1_val = []
F1_test = []
durations = []
best_test_acc = float('-inf')
test_acc_history = []
best_test_f1 = float('-inf')
test_f1_history = []
for epoch in range(300):
    t0 = time.time()
    model.train()
    for it, sg in enumerate(dataloader):
        x = sg.ndata["feat"]
        y = sg.ndata["label"].to(torch.int64)
        m = sg.ndata["train_mask"].bool()
        #x = prompt(it,x)
        y_hat = model(sg, x)
        #y_hat = prompt(it, y_hat)
        loss = F.cross_entropy(y_hat[m], y[m])
        loss = loss + 0.01 * utils.constraint(model.get_prompt())
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
            #x = prompt(it, x)
            y_hat = model(sg, x)
            #y_hat = prompt(it, y_hat)
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
        f1_val = calculate_f1_score(val_preds, val_labels.to('cuda'), num_classes=dataset.num_classes)
        f1_test = calculate_f1_score(test_preds, test_labels.to('cuda'), num_classes=dataset.num_classes)
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    with torch.no_grad():
        graph = graph.to(device)  # Move graph to GPU/CPU
        all_feats = graph.ndata["feat"].to(device)  # Move features to GPU/CPU
        m_test = graph.ndata["test_mask"].bool()
        all_labels = labels.to(device)  # Move labels to GPU/CPU

        # Get node embeddings from the model
        embeddings, pp = model(graph, all_feats)
        embeddings = embeddings[m_test]
        test_labels = all_labels[m_test]

        # Debugging information
        print("Type of embeddings:", type(embeddings))
        print("Embeddings shape:", embeddings.shape)

        # Move embeddings back to CPU and convert to numpy
        embeddings = embeddings.cpu().numpy()

        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=7)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Process prompt embeddings
        class_embeddings = model.linear(pp.detach())
        print("Prompt Embeddings shape:", class_embeddings.shape)
        class_embeddings = class_embeddings.cpu().numpy()
        class_embeddings_2d = tsne.fit_transform(class_embeddings)

        # Debugging information
        print("t-SNE completed")

        # Plotting
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=test_labels.cpu().numpy(), cmap='viridis',
                              alpha=0.7)
        plt.scatter(class_embeddings_2d[:, 0], class_embeddings_2d[:, 1], marker='*', s=200, c='red',
                    label='Prompt Embeddings')
        plt.colorbar(scatter, label='Node Labels')
        plt.title('Node and Prompt Feature Embeddings')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()

        # Save the plot as a PDF
        plt.savefig("com_add_5.pdf", format='pdf')
        plt.show()


        test_labels = test_labels.cpu().numpy()

        # Calculate average embeddings for nodes with the same label
        label_set = np.unique(test_labels)
        label_embeddings = []

        for label in label_set:
            label_indices = np.where(test_labels == label)[0]
            average_embedding = embeddings[label_indices].mean(axis=0)
            label_embeddings.append(average_embedding)

        label_embeddings = np.array(label_embeddings)

        # Plot heatmap of pp

        similarity_matrix = cosine_similarity(class_embeddings, class_embeddings)
        plt.figure(figsize=(12, 8))
        sns.heatmap(similarity_matrix, cmap='YlGnBu', annot=False)
        plt.title('Heatmap of Similarity between Label Embeddings and Prompt Embeddings')
        plt.xlabel('Prompt Embeddings')
        plt.ylabel('Label Embeddings')
        plt.savefig("label_prompt_similarity_heatmap.pdf", format='pdf')
        plt.show()


# Assuming you have already defined `model`, `graph`, and `graph.ndata['label']`


# Assuming you have already defined `model`, `graph`, and `graph.ndata['label']`
plot_embeddings(model, graph, graph.ndata['label'])"""

