import argparse
import os.path as osp
import sys, traceback, pdb

import torch
import torch.nn.functional as F
import os.path as osp
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from baseGNN import GAT
from GINN import GINN
from EGINN import EGINN
from dataset import PDGDataset
from ginnDataset import GINNDataset
from EGINNDataset import EGINNDataset
from config import Config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gnn_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PDGDataset')
ginn_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'GINNDataset')
eginn_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'EGINNDataset')

inputJsonPath_gnnTrain = [["jsondata/intervals-projects-defects4j-train-CFG.json",
                 "jsondata/intervals-projects-defects4j-Array-train-CFG.json",
                 "jsondata/intervals-projects-defects4j-CCE-train-CFG.json"],
                ["jsondata/intervals-projects-dotjar-train-CFG.json",
                 "jsondata/intervals-projects-dotjar-Array-train-CFG.json",
                 "jsondata/intervals-projects-dotjar-CCE-train-CFG.json"],
                ["jsondata/intervals-projects-fse14-train-CFG.json",
                 "jsondata/intervals-projects-fse14-Array-train-CFG.json",
                 "jsondata/intervals-projects-fse14-CCE-train-CFG.json"]]
# gnnTrainDataset = PDGDataset(inputJsonPath, gnn_path, "train.pt")
inputJsonPath_gnnVal = [["jsondata/intervals-projects-defects4j-Array-test-CFG.json",
                 "jsondata/intervals-projects-defects4j-CCE-test-CFG.json",
                 "jsondata/intervals-projects-defects4j-test-CFG.json"],
                ["jsondata/intervals-projects-dotjar-Array-test-CFG.json",
                 "jsondata/intervals-projects-dotjar-CCE-test-CFG.json",
                 "jsondata/intervals-projects-dotjar-test-CFG.json"],
                ["jsondata/intervals-projects-fse14-Array-test-CFG.json",
                 "jsondata/intervals-projects-fse14-CCE-test-CFG.json",
                 "jsondata/intervals-projects-fse14-test-CFG.json"]]
# gnnValDataset = PDGDataset(inputJsonPath, gnn_path, "val.pt")
inputJsonPath_gnnTest = [["jsondata/intervals-projects-defects4j-test-CFG.json"],
                ["jsondata/intervals-projects-dotjar-test-CFG.json"],
                ["jsondata/intervals-projects-fse14-test-CFG.json"]]
# gnnTestDataset = PDGDataset(inputJsonPath, gnn_path, "test.pt")
# gnn_train_loader = DataLoader(gnnTrainDataset, batch_size=10, shuffle=False)
# gnn_val_loader = DataLoader(gnnValDataset, batch_size=10)
# gnn_test_loader = DataLoader(gnnTestDataset, batch_size=10)

inputJsonPath_ginnTrain = [["jsondata/intervals-projects-defects4j-train.json",
                 "jsondata/intervals-projects-defects4j-Array-train.json",
                 "jsondata/intervals-projects-defects4j-CCE-train.json"],
                ["jsondata/intervals-projects-dotjar-train.json",
                 "jsondata/intervals-projects-dotjar-Array-train.json",
                 "jsondata/intervals-projects-dotjar-CCE-train.json"],
                ["jsondata/intervals-projects-fse14-train.json",
                 "jsondata/intervals-projects-fse14-Array-train.json",
                 "jsondata/intervals-projects-fse14-CCE-train.json"]]
# ginnTrainDataset = GINNDataset(inputJsonPath, ginn_path, "train.pt")
inputJsonPath_ginnVal = [["jsondata/intervals-projects-defects4j-Array-test.json",
                 "jsondata/intervals-projects-defects4j-CCE-test.json",
                 "jsondata/intervals-projects-defects4j-test.json"],
                ["jsondata/intervals-projects-dotjar-Array-test.json",
                 "jsondata/intervals-projects-dotjar-CCE-test.json",
                 "jsondata/intervals-projects-dotjar-test.json"],
                ["jsondata/intervals-projects-fse14-Array-test.json",
                 "jsondata/intervals-projects-fse14-CCE-test.json",
                 "jsondata/intervals-projects-fse14-test.json"]]
# ginnValDataset = GINNDataset(inputJsonPath, ginn_path, "val.pt")
inputJsonPath_ginnTest = [["jsondata/intervals-projects-defects4j-test.json"],
                ["jsondata/intervals-projects-dotjar-test.json"],
                ["jsondata/intervals-projects-fse14-test.json"]]
# ginnTestDataset = GINNDataset(inputJsonPath, ginn_path, "test.pt")
# ginn_train_loader = DataLoader(ginnTrainDataset, batch_size=10, shuffle=False)
# ginn_val_loader = DataLoader(ginnValDataset, batch_size=10)
# ginn_test_loader = DataLoader(ginnTestDataset, batch_size=10)

inputJsonPath_eginnTrain = [["jsondata/intervals-projects-defects4j-train-EGINN.json",
                 "jsondata/intervals-projects-defects4j-Array-train-EGINN.json",
                 "jsondata/intervals-projects-defects4j-CCE-train-EGINN.json"],
                ["jsondata/intervals-projects-dotjar-train-EGINN.json",
                 "jsondata/intervals-projects-dotjar-Array-train-EGINN.json",
                 "jsondata/intervals-projects-dotjar-CCE-train-EGINN.json"],
                ["jsondata/intervals-projects-fse14-train-EGINN.json",
                 "jsondata/intervals-projects-fse14-Array-train-EGINN.json",
                 "jsondata/intervals-projects-fse14-CCE-train-EGINN.json"]]
# eginnTrainDataset = EGINNDataset(inputJsonPath, eginn_path, "train.pt")
inputJsonPath_eginnVal = [["jsondata/intervals-projects-defects4j-Array-test-EGINN.json",
                 "jsondata/intervals-projects-defects4j-CCE-test-EGINN.json",
                 "jsondata/intervals-projects-defects4j-test-EGINN.json"],
                ["jsondata/intervals-projects-dotjar-Array-test-EGINN.json",
                 "jsondata/intervals-projects-dotjar-CCE-test-EGINN.json",
                 "jsondata/intervals-projects-dotjar-test-EGINN.json"],
                ["jsondata/intervals-projects-fse14-Array-test-EGINN.json",
                 "jsondata/intervals-projects-fse14-CCE-test-EGINN.json",
                 "jsondata/intervals-projects-fse14-test-EGINN.json"]]
# eginnValDataset = EGINNDataset(inputJsonPath, eginn_path, "val.pt")
inputJsonPath_eginnTest = [["jsondata/intervals-projects-defects4j-test-EGINN.json"],
                ["jsondata/intervals-projects-dotjar-test-EGINN.json"],
                ["jsondata/intervals-projects-fse14-test-EGINN.json"]]
# eginnTestDataset = EGINNDataset(inputJsonPath, eginn_path, "test.pt")
# eginn_train_loader = DataLoader(eginnTrainDataset, batch_size=4, shuffle=False)
# eginn_val_loader = DataLoader(eginnValDataset, batch_size=4)
# eginn_test_loader = DataLoader(eginnTestDataset, batch_size=4)


def create_data_loader(index):
    gnn_train_loader, gnn_val_loader, gnn_test_loader, ginn_train_loader, ginn_val_loader, \
    ginn_test_loader, eginn_train_loader, eginn_val_loader, eginn_test_loader = \
        [None, None, None], [None, None, None], [None, None, None], [None, None, None], [None, None, None], [None, None, None], [None, None, None], [None, None, None], [None, None, None]

    for index in range(3):
        gnnTrainDataset = PDGDataset(inputJsonPath_gnnTrain[index], gnn_path, "train-" + str(index) + ".pt")
        gnnValDataset = PDGDataset(inputJsonPath_gnnVal[index], gnn_path, "val-" + str(index) + ".pt")
        gnnTestDataset = PDGDataset(inputJsonPath_gnnTest[index], gnn_path, "test-" + str(index) + ".pt")
        gnn_train_loader[index] = DataLoader(gnnTrainDataset, batch_size=10, shuffle=False)
        gnn_val_loader[index] = DataLoader(gnnValDataset, batch_size=10)
        gnn_test_loader[index] = DataLoader(gnnTestDataset, batch_size=10)

        ginnTrainDataset = GINNDataset(inputJsonPath_ginnTrain[index], ginn_path, "train-" + str(index) + ".pt")
        ginnValDataset = GINNDataset(inputJsonPath_ginnVal[index], ginn_path, "val-" + str(index) + ".pt")
        ginnTestDataset = GINNDataset(inputJsonPath_ginnTest[index], ginn_path, "test-" + str(index) + ".pt")
        ginn_train_loader[index] = DataLoader(ginnTrainDataset, batch_size=10, shuffle=False)
        ginn_val_loader[index] = DataLoader(ginnValDataset, batch_size=10)
        ginn_test_loader[index] = DataLoader(ginnTestDataset, batch_size=10)

        eginnTrainDataset = EGINNDataset(inputJsonPath_eginnTrain[index], eginn_path, "train-" + str(index) + ".pt")
        eginnValDataset = EGINNDataset(inputJsonPath_eginnVal[index], eginn_path, "val-" + str(index) + ".pt")
        eginnTestDataset = EGINNDataset(inputJsonPath_eginnTest[index], eginn_path, "test-" + str(index) + ".pt")
        eginn_train_loader[index] = DataLoader(eginnTrainDataset, batch_size=4, shuffle=False)
        eginn_val_loader[index] = DataLoader(eginnValDataset, batch_size=4)
        eginn_test_loader[index] = DataLoader(eginnTestDataset, batch_size=4)

    num_features_gnn = gnnTrainDataset.num_features
    num_features_ginn = ginnTrainDataset.num_features
    num_features_eginn = eginnTrainDataset.num_features

    return gnn_train_loader, gnn_val_loader, gnn_test_loader, ginn_train_loader, ginn_val_loader, \
           ginn_test_loader, eginn_train_loader, eginn_val_loader, eginn_test_loader, \
           num_features_gnn, num_features_ginn, num_features_eginn

gnn_train_loader, gnn_val_loader, gnn_test_loader, ginn_train_loader, ginn_val_loader, ginn_test_loader, \
    eginn_train_loader, eginn_val_loader, eginn_test_loader, in_channels_gnn, in_channels_ginn, in_channels_eginn = create_data_loader(0)

def train(model_type, model, train_loader, optimizer):
    model.train()
    total_loss = 0
    center = None
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if model_type == 0:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        elif model_type == 1:
            out = model(data.intra_interval_x, data.intra_interval_edge_index, data.intra_interval_edge_attr, data.inter_interval_edge_index, data.inter_interval_edge_attr, data.intra_interval_node_ids, data.batch)
        else:
            out = model((data.caller_intra_interval_x, data.target_intra_interval_x, data.callee_intra_interval_x), (data.caller_intra_interval_edge_index, data.target_intra_interval_edge_index, data.callee_intra_interval_edge_index), (data.caller_intra_interval_edge_attr, data.target_intra_interval_edge_attr, data.callee_intra_interval_edge_attr), (data.caller_inter_interval_edge_index, data.target_inter_interval_edge_index, data.callee_inter_interval_edge_index),
                    (data.caller_inter_interval_edge_attr, data.target_inter_interval_edge_attr, data.callee_inter_interval_edge_attr), (data.caller_intra_interval_node_ids, data.target_intra_interval_node_ids, data.callee_intra_interval_node_ids), (data.caller_inter_interval_node_ids, data.target_inter_interval_node_ids, data.callee_inter_interval_node_ids), (data.caller_offset, data.target_offset), data.target_entry_node, (data.caller_calling_mask, data.target_calling_mask), data.caller_empty_mask)
        loss = F.cross_entropy(out, data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(model_type, model, loader):
    model.eval()
    corrects, total_ratio = [], 0
    scores, y = [], []
    allRes = []
    groundtruths = []
    for data in loader:
        data = data.to(device)
        if model_type == 0:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        elif model_type == 1:
            out = model(data.intra_interval_x, data.intra_interval_edge_index, data.intra_interval_edge_attr, data.inter_interval_edge_index, data.inter_interval_edge_attr, data.intra_interval_node_ids, data.batch)
        else:
            out = model((data.caller_intra_interval_x, data.target_intra_interval_x, data.callee_intra_interval_x), (data.caller_intra_interval_edge_index, data.target_intra_interval_edge_index, data.callee_intra_interval_edge_index), (data.caller_intra_interval_edge_attr, data.target_intra_interval_edge_attr, data.callee_intra_interval_edge_attr), (data.caller_inter_interval_edge_index, data.target_inter_interval_edge_index, data.callee_inter_interval_edge_index),
                    (data.caller_inter_interval_edge_attr, data.target_inter_interval_edge_attr, data.callee_inter_interval_edge_attr), (data.caller_intra_interval_node_ids, data.target_intra_interval_node_ids, data.callee_intra_interval_node_ids), (data.caller_inter_interval_node_ids, data.target_inter_interval_node_ids, data.callee_inter_interval_node_ids), (data.caller_offset, data.target_offset), data.target_entry_node, (data.caller_calling_mask, data.target_calling_mask), data.caller_empty_mask)
        pred = out.argmax(-1)
        score = out.softmax(-1)
        corrects.append(pred.eq(data.y.to(torch.long)))
        scores.append(score)
        y.append(data.y.to(torch.long))
    return torch.cat(corrects, dim=0), torch.cat(scores, dim=0),torch.cat(y, dim=0) , allRes


def run(model_type, epochs, lr, index):

    if model_type == 0:
        model = GAT(in_channels_gnn, Config.hidden_channels, 2, Config.heads).to(device)
        train_loader = gnn_train_loader[index]
        val_loader = gnn_val_loader[index]
        test_loader = gnn_test_loader[index]
    elif model_type == 1:
        model = GINN(in_channels_ginn, Config.hidden_channels, 2, Config.heads).to(device)
        train_loader = ginn_train_loader[index]
        val_loader = ginn_val_loader[index]
        test_loader = ginn_test_loader[index]
    else:
        model = EGINN(in_channels_eginn, Config.hidden_channels, 2, Config.heads).to(device)
        train_loader = eginn_train_loader[index]
        val_loader = eginn_val_loader[index]
        test_loader = eginn_test_loader[index]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_acc = best_val_auc = 0
    best_val_acc_in_5 = best_val_auc_in_5 = 0
    acc_per_5epoch = []
    auc_per_5epoch = []
    for epoch in range(epochs):
        loss = train(model_type, model, train_loader, optimizer)
        train_correct, _, _, _ = test(model_type, model, train_loader)
        val_correct, val_score, val_y, res = test(model_type, model, val_loader)
        train_acc = train_correct.sum().item() / train_correct.size(0)
        val_acc = val_correct.sum().item() / val_correct.size(0)
        auc_score = roc_auc_score(val_y.numpy(), val_score[:,1].numpy())

        if val_acc > best_val_acc_in_5:
            best_val_acc_in_5 = val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        if auc_score > best_val_auc_in_5:
            best_val_auc_in_5 = auc_score
            if auc_score > best_val_auc:
                best_val_auc = auc_score
        if epoch % 5 == 4:
            acc_per_5epoch.append(best_val_acc_in_5)
            auc_per_5epoch.append(best_val_auc_in_5)
            best_val_acc_in_5 = best_val_auc_in_5 = 0
    return best_val_acc, best_val_auc, acc_per_5epoch, auc_per_5epoch

def main():
    model_name = ["GNN", "GINN", "EGINN"]
    dataset_name = ["defects4j", "dotjar", "fse14"]

    def run_for_epochs_curve():
        gnn_acc, gnn_auc, gnn_acc_per5, gnn_auc_per5 = run(model_type=0, epochs=200, lr=0.005, index=0)
        ginn_acc, ginn_auc, ginn_acc_per5, ginn_auc_per5 = run(model_type=1, epochs=200, lr=0.005, index=0)
        eginn_acc, eginn_auc, eginn_acc_per5, eginn_auc_per5 = run(model_type=2, epochs=200, lr=0.005, index=0)
        x = [i for i in range(5, 201, 5)]
        # auc curves
        plt.plot(x, gnn_auc_per5, label='GNN')
        plt.plot(x, ginn_auc_per5, label='GINN')
        plt.plot(x, eginn_auc_per5, label='EGINN')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.savefig('figs/AUC_Epoch_curve.png')
        plt.cla()
        # acc curves
        plt.plot(x, gnn_acc_per5, label='GNN')
        plt.plot(x, ginn_acc_per5, label='GINN')
        plt.plot(x, eginn_acc_per5, label='EGINN')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('figs/Accuracy_Epoch_curve.png')
        plt.cla()

        with open("out/200-epochs.txt", "w") as f:
            f.write(f'GNN best accuracy: {gnn_acc}, auc: {gnn_auc}\n')
            f.write(f'GINN best accuracy: {ginn_acc}, auc: {ginn_auc}\n')
            f.write(f'EGINN best accuracy: {eginn_acc}, auc: {eginn_auc}\n')
        return gnn_acc_per5, gnn_auc_per5, ginn_acc_per5, ginn_auc_per5, eginn_acc_per5, eginn_auc_per5

    caption = "performance on epochs curve"
    gnn_acc5, gnn_auc5, ginn_acc5, ginn_auc5, eginn_acc5, eginn_auc5 = run_for_epochs_curve()

    def run_for_each_dataset():
        GNN_acc = GINN_acc = EGINN_acc = []
        GNN_auc = GINN_auc = EGINN_auc = []
        for i in range(3):
            gnn_acc, gnn_auc, _, _ = run(model_type=0, epochs=50, lr=0.005, index=i)
            ginn_acc, ginn_auc, _, _ = run(model_type=1, epochs=50, lr=0.005, index=i)
            eginn_acc, eginn_auc, _, _ = run(model_type=2, epochs=50, lr=0.005, index=i)
            GNN_acc.append(gnn_acc)
            GNN_auc.append(gnn_auc)
            GINN_acc.append(ginn_acc)
            GINN_auc.append(ginn_auc)
            EGINN_acc.append(eginn_acc)
            EGINN_auc.append(eginn_auc)
        with open("out/performance-on-each-dataset.txt", "w") as f:
            for i in range(3):
                f.write(f'  Performance on {dataset_name[i]}:\n')
                f.write(f'GNN best accuracy: {GNN_acc[i]}, auc: {GNN_auc[i]}\n')
                f.write(f'GINN best accuracy: {GINN_acc[i]}, auc: {GINN_auc[i]}\n')
                f.write(f'EGINN best accuracy: {EGINN_acc[i]}, auc: {EGINN_auc[i]}\n\n')

    caption = "performance on different dataset"
    run_for_each_dataset()

    def run_for_different_lr():
        GNN_acc_per5 = GINN_acc_per5 = EGINN_acc_per5 = []
        GNN_auc_per5 = GINN_auc_per5 = EGINN_auc_per5 = []
        gnn_acc, gnn_auc, gnn_acc_per5, gnn_auc_per5 = run(model_type=0, epochs=200, lr=0.001, index=0)
        ginn_acc, ginn_auc, ginn_acc_per5, ginn_auc_per5 = run(model_type=1, epochs=200, lr=0.001, index=0)
        eginn_acc, eginn_auc, eginn_acc_per5, eginn_auc_per5 = run(model_type=2, epochs=200, lr=0.001, index=0)
        GNN_acc_per5.append(gnn_acc_per5)
        GINN_acc_per5.append(ginn_acc_per5)
        EGINN_acc_per5.append(eginn_acc_per5)
        GNN_auc_per5.append(gnn_auc_per5)
        GINN_auc_per5.append(ginn_auc_per5)
        EGINN_auc_per5.append(eginn_auc_per5)
        GNN_acc_per5.append(gnn_acc5)
        GINN_acc_per5.append(ginn_acc5)
        EGINN_acc_per5.append(eginn_acc5)
        GNN_auc_per5.append(gnn_auc5)
        GINN_auc_per5.append(ginn_auc5)
        EGINN_auc_per5.append(eginn_auc5)
        gnn_acc, gnn_auc, gnn_acc_per5, gnn_auc_per5 = run(model_type=0, epochs=200, lr=0.01, index=0)
        ginn_acc, ginn_auc, ginn_acc_per5, ginn_auc_per5 = run(model_type=1, epochs=200, lr=0.01, index=0)
        eginn_acc, eginn_auc, eginn_acc_per5, eginn_auc_per5 = run(model_type=2, epochs=200, lr=0.01, index=0)
        GNN_acc_per5.append(gnn_acc_per5)
        GINN_acc_per5.append(ginn_acc_per5)
        EGINN_acc_per5.append(eginn_acc_per5)
        GNN_auc_per5.append(gnn_auc_per5)
        GINN_auc_per5.append(ginn_auc_per5)
        EGINN_auc_per5.append(eginn_auc_per5)
        x = [i for i in range(5, 201, 5)]
        # auc curves
        plt.plot(x, GNN_auc_per5[0], label='l=0.001')
        plt.plot(x, GNN_auc_per5[1], label='l=0.005')
        plt.plot(x, GNN_auc_per5[2], label='l=0.01')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('GNN')
        plt.legend()
        plt.savefig('figs/GNN_AUC_curve.png')
        plt.cla()
        # acc curves
        plt.plot(x, GNN_acc_per5[0], label='l=0.001')
        plt.plot(x, GNN_acc_per5[1], label='l=0.005')
        plt.plot(x, GNN_acc_per5[2], label='l=0.01')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('GNN')
        plt.legend()
        plt.savefig('figs/GNN_Accuracy_curve.png')
        plt.cla()

        # auc curves
        plt.plot(x, GINN_auc_per5[0], label='l=0.001')
        plt.plot(x, GINN_auc_per5[1], label='l=0.005')
        plt.plot(x, GINN_auc_per5[2], label='l=0.01')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('GINN')
        plt.legend()
        plt.savefig('figs/GINN_AUC_curve.png')
        plt.cla()
        # acc curves
        plt.plot(x, GINN_acc_per5[0], label='l=0.001')
        plt.plot(x, GINN_acc_per5[1], label='l=0.005')
        plt.plot(x, GINN_acc_per5[2], label='l=0.01')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('GINN')
        plt.legend()
        plt.savefig('figs/GINN_Accuracy_curve.png')
        plt.cla()

        # auc curves
        plt.plot(x, EGINN_auc_per5[0], label='l=0.001')
        plt.plot(x, EGINN_auc_per5[1], label='l=0.005')
        plt.plot(x, EGINN_auc_per5[2], label='l=0.01')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('EGINN')
        plt.legend()
        plt.savefig('figs/EGINN_AUC_curve.png')
        plt.cla()
        # acc curves
        plt.plot(x, EGINN_acc_per5[0], label='l=0.001')
        plt.plot(x, EGINN_acc_per5[1], label='l=0.005')
        plt.plot(x, EGINN_acc_per5[2], label='l=0.01')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('EGINN')
        plt.legend()
        plt.savefig('figs/EGINN_Accuracy_curve.png')
        plt.cla()

    run_for_different_lr()

if __name__ == '__main__':
    try:
        main()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)