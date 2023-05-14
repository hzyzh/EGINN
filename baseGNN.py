import argparse
import os.path as osp
import sys, traceback, pdb

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv, TopKPooling, global_mean_pool

from dataset import PDGDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=8)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'GAT-{args.dataset}', heads=args.heads, epochs=args.epochs,
           hidden_channels=args.hidden_channels, lr=args.lr, device=device)

#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
#dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PDGDataset')
inputJsonPath = "./intervals-projects-defects4j-train-CFG.json"
trainDataset = PDGDataset(inputJsonPath, path, "train.pt")
inputJsonPath = "./intervals-projects-defects4j-val-CFG.json"
valDataset = PDGDataset(inputJsonPath, path, "val.pt")
inputJsonPath = "./intervals-projects-defects4j-test-CFG.json"
testDataset = PDGDataset(inputJsonPath, path, "test.pt")
#FIXME: shuffle=false and batchsize may be larger
train_loader = DataLoader(trainDataset, batch_size=10, shuffle=False)
val_loader = DataLoader(valDataset, batch_size=10)
test_loader = DataLoader(testDataset, batch_size=10)
#data = dataset[0].to(device)
#data.train_mask

class GAT(torch.nn.Module):
    rep_dim = 0
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, 16, heads=1,
                             concat=False, dropout=0.6)
        self.fc = torch.nn.Linear(16, out_channels)
        self.rep_dim = out_channels

    def message_passing(self, x, edge_index, edge_attr, nodeIDs):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        # TODO: maybe pick the top-K nodes for aggreating graph representations.
        return x

    def forward(self, x, edge_index, edge_attr, nodeIDs):
        x = self.message_passing(x, edge_index, edge_attr, nodeIDs)
        x = global_mean_pool(x, nodeIDs)
        return self.fc(x)


model = GAT(trainDataset.num_features, args.hidden_channels, trainDataset.num_classes,
            args.heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def init_center_c(train_loader, net, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(net.rep_dim, device=device)

    net.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            data = data.to(device)
            outputs = net(data.x, data.edge_index, data.edge_attr, data.batch)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

def train():
    model.train()
    total_loss = 0
    center = None
    center = init_center_c(train_loader, model)
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.cross_entropy(out, data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


def inteprePrediction(preds, topK):
    center = torch.mean(preds)
    dist = torch.sum((preds - center) ** 2, dim=1)
    ind = torch.topk(dist, topK).indices
    res = torch.zeros(preds.shape[0], dtype=torch.int)
    res[ind] = 1
    return res

@torch.no_grad()
def test(loader):
    model.eval()
    corrects, total_ratio = [], 0
    allRes = []
    groundtruths = []
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.edge_attr, data.batch).argmax(dim=-1)
        corrects.append(pred.eq(data.y.to(torch.long)))
    return torch.cat(corrects, dim=0), allRes


def run():
    best_val_acc = final_test_acc = 0
    test_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train()
        train_correct, _ = test(train_loader)
        val_correct, _ = test(val_loader)
        test_correct, res = test(test_loader)
        train_acc = train_correct.sum().item() / train_correct.size(0)
        val_acc = val_correct.sum().item() / val_correct.size(0)
        tmp_test_acc = test_correct.sum().item() / test_correct.size(0)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)

if __name__ == '__main__':
    try:
        run()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
