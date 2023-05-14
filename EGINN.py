import argparse
import os.path as osp
import sys, traceback, pdb

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv, TopKPooling, global_mean_pool
from torch_geometric.nn.pool import avg_pool

from EGINNDataset import EGINNDataset
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
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'EGINNDataset')
# inputJsonPath = "./eintervals-example.json"
inputJsonPath = ["jsondata/intervals-projects-defects4j-train-EGINN.json",
                 "jsondata/intervals-projects-dotjar-train-EGINN.json",
                 "jsondata/intervals-projects-fse14-train-EGINN.json",
                 ]
trainDataset = EGINNDataset(inputJsonPath, path, "train.pt")
inputJsonPath = ["jsondata/intervals-projects-defects4j-test-EGINN.json",
                 "jsondata/intervals-projects-dotjar-test-EGINN.json",
                 "jsondata/intervals-projects-fse14-test-EGINN.json"]
valDataset = EGINNDataset(inputJsonPath, path, "val.pt")
testDataset = EGINNDataset(inputJsonPath, path, "test.pt")
#FIXME: shuffle=false and batchsize may be larger

train_loader = DataLoader(trainDataset, batch_size=4, shuffle=False)
val_loader = DataLoader(valDataset, batch_size=4)
test_loader = DataLoader(testDataset, batch_size=4)
#data = dataset[0].to(device)
#data.train_mask

class EGINN(torch.nn.Module):
    rep_dim = 0
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        # TODO: replace it with a better one
        self.gatIntra = GAT(in_channels, hidden_channels, out_channels, heads)
        self.gatInter = GAT(16, hidden_channels, out_channels, heads)
        self.rep_dim = out_channels
        self.gatCaller = nn.Sequential(
            nn.Linear(16 * 2, hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels * 2, 16),
            nn.ReLU(inplace=True),
        )
        self.gatCallee = nn.Sequential(
            nn.Linear(16 * 2, hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels * 2, 16),
            nn.ReLU(inplace=True),
        )


    def forward(self, intra_interval_x, intra_interval_edge_index, intra_interval_edge_attr, inter_interval_edge_index, inter_interval_edge_attr, intra_interval_node_ids, inter_interval_node_ids, offset, target_entry_node, calling_mask, caller_empty_mask):

        caller_intra_interval_x, target_intra_interval_x, callee_intra_interval_x = intra_interval_x
        caller_intra_interval_edge_index, target_intra_interval_edge_index, callee_intra_interval_edge_index = intra_interval_edge_index
        caller_intra_interval_edge_attr, target_intra_interval_edge_attr, callee_intra_interval_edge_attr = intra_interval_edge_attr
        caller_inter_interval_edge_index, target_inter_interval_edge_index, callee_inter_interval_edge_index = inter_interval_edge_index
        caller_inter_interval_edge_attr, target_inter_interval_edge_attr, callee_inter_interval_edge_attr = inter_interval_edge_attr
        caller_intra_interval_node_ids, target_intra_interval_node_ids, callee_intra_interval_node_ids = intra_interval_node_ids
        caller_inter_interval_node_ids, target_inter_interval_node_ids, callee_inter_interval_node_ids = inter_interval_node_ids
        caller_offset, target_offset = offset
        caller_calling_mask, target_calling_mask = calling_mask

        # process callers
        process_caller_flag = len(caller_intra_interval_x) > 0
        if process_caller_flag:
            caller_intra_interval_x = self.gatIntra.message_passing(caller_intra_interval_x,
                                                                    caller_intra_interval_edge_index,
                                                                    caller_intra_interval_edge_attr,
                                                                    caller_intra_interval_node_ids)
            for i in range(2):
                caller_intra_interval_x = self.gatInter.message_passing(caller_intra_interval_x,
                                                                        caller_intra_interval_edge_index,
                                                                        caller_intra_interval_edge_attr,
                                                                        caller_intra_interval_node_ids)
                # the second arg of global_mean_pool:
                caller_inter_interval_x = global_mean_pool(caller_intra_interval_x, caller_intra_interval_node_ids)
                caller_inter_interval_x = self.gatInter.message_passing(caller_inter_interval_x,
                                                                        caller_inter_interval_edge_index,
                                                                        caller_inter_interval_edge_attr,
                                                                        caller_inter_interval_node_ids)
                caller_intra_interval_x = torch.index_select(caller_inter_interval_x, 0, caller_intra_interval_node_ids)

            caller_inter_interval_x = global_mean_pool(caller_inter_interval_x, caller_inter_interval_node_ids)
        else:
            caller_inter_interval_x = torch.tensor([])

        caller_entry_ids = torch.index_select(target_entry_node, 0, caller_offset)


        # process callees
        process_callee_flag = len(callee_intra_interval_x) > 0
        if process_callee_flag:
            callee_intra_interval_x = self.gatIntra.message_passing(callee_intra_interval_x,
                                                                    callee_intra_interval_edge_index,
                                                                    callee_intra_interval_edge_attr,
                                                                    callee_intra_interval_node_ids)
            for i in range(2):
                callee_intra_interval_x = self.gatInter.message_passing(callee_intra_interval_x,
                                                                        callee_intra_interval_edge_index,
                                                                        callee_intra_interval_edge_attr,
                                                                        callee_intra_interval_node_ids)
                # the second arg of global_mean_pool:
                callee_inter_interval_x = global_mean_pool(callee_intra_interval_x, callee_intra_interval_node_ids)
                callee_inter_interval_x = self.gatInter.message_passing(callee_inter_interval_x,
                                                                        callee_inter_interval_edge_index,
                                                                        callee_inter_interval_edge_attr,
                                                                        callee_inter_interval_node_ids)
                callee_intra_interval_x = torch.index_select(callee_inter_interval_x, 0, callee_intra_interval_node_ids)

            callee_inter_interval_x = global_mean_pool(callee_inter_interval_x, callee_inter_interval_node_ids)
        else:
            callee_inter_interval_x = torch.tensor([])

        target_nodes_num = len(target_intra_interval_x)
        callee_mask = torch.zeros(len(callee_inter_interval_x), target_nodes_num).bool()
        all_mask = torch.zeros(target_nodes_num).bool()

        node_index_offset = 0
        for target_index in range(len(target_calling_mask)):
                tcm = target_calling_mask[target_index]
                for i in range(len(tcm)):
                    if len(tcm[i]) > 0:
                        all_mask[i + node_index_offset] = True
                    for callee_id in tcm[i]:
                        callee_mask[callee_id + target_offset[target_index]][i + node_index_offset] = True
                node_index_offset += len(tcm)


        # process target
        target_interprocedural_message = self.gatIntra.message_passing(target_intra_interval_x,
                                                                target_intra_interval_edge_index,
                                                                target_intra_interval_edge_attr,
                                                                target_intra_interval_node_ids)
        for i in range(2):
            # process target
            target_intra_interval_x = target_interprocedural_message
            target_intra_interval_x = self.gatInter.message_passing(target_intra_interval_x,
                                                                    target_intra_interval_edge_index,
                                                                    target_intra_interval_edge_attr,
                                                                    target_intra_interval_node_ids)
            # the second arg of global_mean_pool:
            target_inter_interval_x = global_mean_pool(target_intra_interval_x, target_intra_interval_node_ids)
            target_inter_interval_x = self.gatInter.message_passing(target_inter_interval_x,
                                                                    target_inter_interval_edge_index,
                                                                    target_inter_interval_edge_attr,
                                                                    target_inter_interval_node_ids)
            target_intra_interval_x = torch.index_select(target_inter_interval_x, 0, target_intra_interval_node_ids)
            # interprocedural message passing:
            # ------------------------------------------------------------------------------
            # caller interprocedural message passing
            if process_caller_flag:
                entry_node_x_per_caller = torch.index_select(target_intra_interval_x, 0, caller_entry_ids)
                caller_interprocedural_message = torch.column_stack((caller_inter_interval_x, entry_node_x_per_caller))
                caller_interprocedural_message = self.gatCaller(caller_interprocedural_message)
                caller_interprocedural_message = global_mean_pool(caller_interprocedural_message, caller_offset,
                                                                  len(target_offset))

                entry_node_x = torch.index_select(target_intra_interval_x, 0, target_entry_node)
                entry_node_x_without_caller = entry_node_x.masked_fill(~caller_empty_mask, 0)
                entry_node_x_with_caller = entry_node_x.masked_fill(caller_empty_mask, 0)
                caller_interprocedural_message = torch.add(caller_interprocedural_message, entry_node_x_without_caller)
                caller_interprocedural_masked_message = global_mean_pool(caller_interprocedural_message, target_entry_node,
                                                                         len(target_intra_interval_x))
                entry_node_x_with_caller = global_mean_pool(entry_node_x_with_caller, target_entry_node,
                                                            len(target_intra_interval_x))

                target_interprocedural_message = torch.add(target_intra_interval_x, entry_node_x_with_caller, alpha=-0.5)
                target_interprocedural_message = torch.add(target_interprocedural_message,
                                                           caller_interprocedural_masked_message, alpha=0.5)
            else:
                target_interprocedural_message = target_intra_interval_x
            # ------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------
            # callee interprocedural message passing
            if process_callee_flag:
                callee_interprocedural_message_in_all = torch.zeros_like(target_interprocedural_message)
                for callee_index in range(len(callee_inter_interval_x)):
                    target_intra_interval_x_masked = target_interprocedural_message.masked_select(
                        callee_mask[callee_index].reshape(target_nodes_num, 1))
                    target_intra_interval_x_masked = target_intra_interval_x_masked.reshape(
                        -1, target_interprocedural_message.shape[-1])
                    callee_interprocedural_message = callee_inter_interval_x[callee_index].repeat(
                        len(target_intra_interval_x_masked), 1)
                    callee_interprocedural_message = torch.column_stack(
                        (callee_interprocedural_message, target_intra_interval_x_masked))
                    callee_interprocedural_message = self.gatCallee(callee_interprocedural_message)
                    interprocedural_message_per_callee = target_interprocedural_message.clone()
                    interprocedural_message_per_callee[callee_mask[callee_index]] = callee_interprocedural_message
                    callee_interprocedural_message_in_all = callee_interprocedural_message_in_all.add(
                        interprocedural_message_per_callee)

                target_interprocedural_message = callee_interprocedural_message_in_all.div(
                    len(callee_inter_interval_x))
            # ------------------------------------------------------------------------------

        target_inter_interval_x = global_mean_pool(target_inter_interval_x, target_inter_interval_node_ids)
        return self.gatInter.fc(target_inter_interval_x)


model = EGINN(trainDataset.num_features, args.hidden_channels, trainDataset.num_classes,
            args.heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
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

def debug():
    for data in train_loader:
        data = data.to(device)
        print(f'caller offset: {data.caller_offset}')
        print(f'target offset: {data.target_offset}')
        print(f'entry node: {data.target_entry_node}')
        print(f'caller intra interval ids: {data.caller_intra_interval_node_ids}')
        print(f'target intra interval ids: {data.target_intra_interval_node_ids}')
        print(f'caller interval ids: {data.caller_inter_interval_node_ids}')
        print(f'target interval ids: {data.target_inter_interval_node_ids}')
        print(f'callee interval ids: {data.callee_inter_interval_node_ids}')
        print(f'caller empty mask: {data.caller_empty_mask}')
        print(f'caller calling mask: {data.caller_calling_mask}')
        print(f'target calling mask: {data.target_calling_mask}')
'''
        for index in range(len(data.caller_calling_mask)):
            mask_per_caller = data.caller_calling_mask[index]
            for cm in mask_per_caller:
                if len(cm) != 0:
                    print(f'call mask: {cm[0] + data.caller_offset[index]}')

        for index in range(len(data.target_calling_mask)):
            mask_per_target = data.target_calling_mask[index]
            for cm in mask_per_target:
                if len(cm) != 0:
                    for m in cm:
                        print(f'target mask: {m + data.target_offset[index]}')
'''
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
        # debug()
        run()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
