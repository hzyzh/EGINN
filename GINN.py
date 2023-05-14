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

from ginnDataset import GINNDataset
from baseGNN import GAT

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
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'GINNDataset')
inputJsonPath = ["jsondata/intervals-projects-defects4j-train.json",
                 "jsondata/intervals-projects-dotjar-train.json",
                 "jsondata/intervals-projects-fse14-train.json",
                 ]
trainDataset = GINNDataset(inputJsonPath, path, "train.pt")
# inputJsonPath = "./intervals-projects-defects4j-test.json"
inputJsonPath = ["jsondata/intervals-projects-defects4j-test.json",
                 "jsondata/intervals-projects-dotjar-test.json",
                 "jsondata/intervals-projects-fse14-test.json"]
valDataset = GINNDataset(inputJsonPath, path, "val.pt")
testDataset = GINNDataset(inputJsonPath, path, "test.pt")
#FIXME: shuffle=false and batchsize may be larger

train_loader = DataLoader(trainDataset, batch_size=10, shuffle=False)
val_loader = DataLoader(valDataset, batch_size=10)
test_loader = DataLoader(testDataset, batch_size=10)
#data = dataset[0].to(device)
#data.train_mask

class GINN(torch.nn.Module):
    rep_dim = 0
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        # TODO: replace it with a better one
        self.gatIntra = GAT(in_channels, hidden_channels, out_channels, heads)
        self.gatInter = GAT(16, hidden_channels, out_channels, heads)
        self.rep_dim = out_channels


    def forward(self, intra_interval_x, intra_interval_edge_index, intra_interval_edge_attr, inter_interval_edge_index, inter_interval_edge_attr, intra_interval_node_ids, inter_interval_node_ids):
        intra_interval_x = self.gatIntra.message_passing(intra_interval_x, intra_interval_edge_index, intra_interval_edge_attr, intra_interval_node_ids)
        for i in range(2):
            intra_interval_x = self.gatInter.message_passing(intra_interval_x, intra_interval_edge_index, intra_interval_edge_attr, intra_interval_node_ids)
            # the second arg of global_mean_pool:
            inter_interval_x = global_mean_pool(intra_interval_x, intra_interval_node_ids)
            inter_interval_x = self.gatInter.message_passing(inter_interval_x, inter_interval_edge_index, inter_interval_edge_attr, inter_interval_node_ids)
            intra_interval_x = torch.index_select(inter_interval_x, 0, intra_interval_node_ids)

        inter_interval_x = global_mean_pool(inter_interval_x, inter_interval_node_ids)
        return self.gatInter.fc(inter_interval_x)


model = GINN(trainDataset.num_features, args.hidden_channels, trainDataset.num_classes,
            args.heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
# print(trainDataset.num_features)

def train():
    model.train()
    total_loss = 0
    center = None
    #center = init_center_c(train_loader, model)
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.intra_interval_x, data.intra_interval_edge_index, data.intra_interval_edge_attr, data.inter_interval_edge_index, data.inter_interval_edge_attr, data.intra_interval_node_ids, data.batch)
        loss = F.cross_entropy(out, data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    corrects, total_ratio = [], 0
    allRes = []
    groundtruths = []
    for data in loader:
        data = data.to(device)
        pred = model(data.intra_interval_x, data.intra_interval_edge_index, data.intra_interval_edge_attr, data.inter_interval_edge_index, data.inter_interval_edge_attr, data.intra_interval_node_ids, data.batch).argmax(-1)
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

def debug():
    for data in train_loader:
        data = data.to(device)
        print(f'node ids: {data.intra_interval_node_ids}')
        print(f'edge index: {data.intra_interval_edge_index}')
        print(f'batch: {data.batch}')

if __name__ == '__main__':
    try:
        # debug()
        run()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
