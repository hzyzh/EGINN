import argparse
import os.path as osp
import sys, traceback, pdb

import torch
import torch.nn.functional as F
import os.path as osp
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from baseGNN import GAT
from GINN import GINN
from EGINN import EGINN
from dataset import PDGDataset
from ginnDataset import GINNDataset
from EGINNDataset import EGINNDataset
from EGINN import EGINN
from config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, nargs='+', help='TRAIN: path to training dataset')
    parser.add_argument('--test', nargs='*', help='TEST: path to training dataset. Default value is set to be the same as training dataset')
    args = parser.parse_args()


    #path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    #dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'EGINNDataset')
    trainDataset = EGINNDataset(args.train, path, "train.pt")
    inputJsonPath = None
    if args.test == None:
        inputJsonPath = args.train
    else:
        inputJsonPath = args.test
    #print(inputJsonPath)
    valDataset = EGINNDataset(inputJsonPath, path, "val.pt")
    testDataset = EGINNDataset(inputJsonPath, path, "test.pt")
    #FIXME: shuffle=false and batchsize may be larger

    train_loader = DataLoader(trainDataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(valDataset, batch_size=4)
    test_loader = DataLoader(testDataset, batch_size=4)
    #data = dataset[0].to(device)
    #data.train_mask

    model = EGINN(trainDataset.num_features, Config.hidden_channels, trainDataset.num_classes,
                Config.heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    # print(trainDataset.num_classes)

def train():
    model.train()
    total_loss = 0
    center = None
    #center = init_center_c(train_loader, model)
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model((data.caller_intra_interval_x, data.target_intra_interval_x, data.callee_intra_interval_x), (data.caller_intra_interval_edge_index, data.target_intra_interval_edge_index, data.callee_intra_interval_edge_index), (data.caller_intra_interval_edge_attr, data.target_intra_interval_edge_attr, data.callee_intra_interval_edge_attr), (data.caller_inter_interval_edge_index, data.target_inter_interval_edge_index, data.callee_inter_interval_edge_index),
                    (data.caller_inter_interval_edge_attr, data.target_inter_interval_edge_attr, data.callee_inter_interval_edge_attr), (data.caller_intra_interval_node_ids, data.target_intra_interval_node_ids, data.callee_intra_interval_node_ids), (data.caller_inter_interval_node_ids, data.target_inter_interval_node_ids, data.callee_inter_interval_node_ids), (data.caller_offset, data.target_offset), data.target_entry_node, (data.caller_calling_mask, data.target_calling_mask), data.caller_empty_mask)
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
        pred = model((data.caller_intra_interval_x, data.target_intra_interval_x, data.callee_intra_interval_x), (data.caller_intra_interval_edge_index, data.target_intra_interval_edge_index, data.callee_intra_interval_edge_index), (data.caller_intra_interval_edge_attr, data.target_intra_interval_edge_attr, data.callee_intra_interval_edge_attr),
                     (data.caller_inter_interval_edge_index, data.target_inter_interval_edge_index, data.callee_inter_interval_edge_index), (data.caller_inter_interval_edge_attr, data.target_inter_interval_edge_attr, data.callee_inter_interval_edge_attr), (data.caller_intra_interval_node_ids, data.target_intra_interval_node_ids, data.callee_intra_interval_node_ids),
                     (data.caller_inter_interval_node_ids, data.target_inter_interval_node_ids, data.callee_inter_interval_node_ids), (data.caller_offset, data.target_offset), data.target_entry_node, (data.caller_calling_mask, data.target_calling_mask), data.caller_empty_mask).argmax(-1)
        corrects.append(pred.eq(data.y.to(torch.long)))
    return torch.cat(corrects, dim=0), allRes


def run():
    best_val_acc = final_test_acc = 0
    test_acc = 0
    for epoch in range(1,  201):
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
        # debug()
        run()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
