import torch
import shutil, json, os
from glob import glob
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from dataset import PDGDataset
from enum import Enum
import numpy as np



class EGINNData(Data):
    def __init__(self, intra_interval_x=(None, None, None), intra_interval_edge_index=(None, None, None), intra_interval_edge_attr=(None, None, None), intra_interval_node_ids=(None, None, None), inter_interval_node_ids=(None, None, None), offset=(None, None), interval_offset=(None, None, None), entry_node=None):
        super().__init__()
        self.caller_intra_interval_x = intra_interval_x[0]
        self.target_intra_interval_x = intra_interval_x[1]
        self.callee_intra_interval_x = intra_interval_x[2]
        self.caller_intra_interval_edge_index = intra_interval_edge_index[0]
        self.target_intra_interval_edge_index = intra_interval_edge_index[1]
        self.callee_intra_interval_edge_index = intra_interval_edge_index[2]
        self.caller_intra_interval_edge_attr = intra_interval_edge_attr[0]
        self.target_intra_interval_edge_attr = intra_interval_edge_attr[1]
        self.callee_intra_interval_edge_attr = intra_interval_edge_attr[2]
        self.caller_intra_interval_node_ids = intra_interval_node_ids[0]
        self.target_intra_interval_node_ids = intra_interval_node_ids[1]
        self.callee_intra_interval_node_ids = intra_interval_node_ids[2]
        self.caller_inter_interval_node_ids = inter_interval_node_ids[0]
        self.target_inter_interval_node_ids = inter_interval_node_ids[1]
        self.callee_inter_interval_node_ids = inter_interval_node_ids[2]
        self.target_entry_node = entry_node
        self.caller_offset = offset[0]
        self.target_offset = offset[1]
        self.caller_interval_offset = interval_offset[0]
        self.target_interval_offset = interval_offset[1]
        self.callee_interval_offset = interval_offset[2]
        self.caller_inter_interval_edge_attr = None
        self.target_inter_interval_edge_attr = None
        self.callee_inter_interval_edge_attr = None
        self.caller_inter_interval_edge_index = None
        self.target_inter_interval_edge_index = None
        self.callee_inter_interval_edge_index = None
        self.caller_num = None
        self.callee_num = None
        self.caller_calling_mask = None
        self.target_calling_mask = None
        self.caller_num_nodes = None
        self.target_num_nodes = None
        self.callee_num_nodes = None
        self.caller_empty_mask = None

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'caller_intra_interval_node_ids':
            return self.caller_interval_offset
        if key == 'target_intra_interval_node_ids':
            return self.target_interval_offset
        if key == 'callee_intra_interval_node_ids':
            return self.callee_interval_offset
        if key == 'caller_inter_interval_node_ids':
            return self.caller_num
        if key == 'target_inter_interval_node_ids':
            return 1
        if key == 'callee_inter_interval_node_ids':
            return self.callee_num
        if key == 'caller_intra_interval_edge_index':
            return self.caller_intra_interval_x.size(0)
        if key == 'target_intra_interval_edge_index':
            return self.target_intra_interval_x.size(0)
        if key == 'callee_intra_interval_edge_index':
            return self.callee_intra_interval_x.size(0)
        if key == "target_entry_node":
            return self.target_intra_interval_x.size(0)
        if key == 'caller_offset':
            return 1
        if key == 'target_offset':
            return self.callee_num
        if key == 'caller_inter_interval_edge_index':
            return self.caller_num_nodes
        if key == 'target_inter_interval_edge_index':
            return self.target_num_nodes
        if key == 'callee_inter_interval_edge_index':
            return self.callee_num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)

class EGINNDataset(InMemoryDataset):
    urlOrPath = "./input.json"
    savedFileName = "train.pt"
    def __init__(self, inputJsonPath, root, savedFileName, transform=None, pre_transform=None, pre_filter=None):
        self.inputJsonPath = inputJsonPath
        self.savedFileName = savedFileName
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.urlOrPath

    @property
    def processed_file_names(self):
        return self.savedFileName

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        jsonData = []
        for fileName in self.inputJsonPath:
            with open(fileName, 'r') as f:
                jsonData += json.load(f)
        self.urlOrPath = os.path.join(self.raw_dir, "EGINN-"+self.savedFileName+".json")

        with open(self.urlOrPath, "w") as outfile:
            json.dump(jsonData, outfile)

    def graph_to_COO(self, graphs, numOfNode, offset=0):
        row = []
        col = []
        edge_feature = []
        for i in range(len(graphs)):
            graph = graphs[i]
            for edge in graph:
                row.append(edge[0]+offset)
                col.append(edge[0]+offset)
                edge_feature.append([edge[1]])
            offset += numOfNode[i]
        edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0).long()
        edge_feature = torch.tensor(edge_feature).float()
        return edge_index, edge_feature, offset

    def processIntervalGraph(self, callers, target, callees, entryNode):
        caller_x, target_x, callee_x = [], [], []
        caller_edge_index, target_edge_index, callee_edge_index = [[], []], [[], []], [[], []]
        caller_edge_feature, target_edge_feature, callee_edge_feature = [], [], []
        caller_node_ids, target_node_ids, callee_node_ids = [], [], []
        caller_interval_node_ids, target_interval_node_ids, callee_interval_node_ids = [], [], []
        caller_calling_mask, target_calling_mask = [], []
        caller_offset = []
        target_offset = 0
        caller_empty_mask = [[False]]
        caller_interval_offset, target_interval_offset, callee_interval_offset = 0, 0, 0
        entry_node = 0

        offset = 0
        interval_offset = 0
        callerNum = len(callers)
        graph_id = 0
        for intervals in callers:
            numofIntervals = len(intervals)
            for index in range(numofIntervals):
                d = intervals[index]
                numOfNodes = len(d["intra_interval_x"])
                for edge in d["graph"]:
                    caller_edge_index[0].append(edge[0]+offset)
                    caller_edge_index[1].append(edge[2]+offset)
                    caller_edge_feature.append([edge[1]])

                caller_x += d["intra_interval_x"]
                caller_node_ids += [index+interval_offset for i in range(numOfNodes)]
                caller_calling_mask += [d["calling_mask"]]
                caller_interval_node_ids += [graph_id]
                offset += numOfNodes

            caller_offset += [0]
            interval_offset += numofIntervals
            graph_id += 1
        caller_interval_offset = interval_offset
        if callerNum == 0:
            caller_empty_mask[0][0] = True

        offset = 0
        assert len(entryNode) > 0
        for index in range(len(target)):
            d = target[index]
            numOfNodes = len(d["intra_interval_x"])
            for edge in d["graph"]:
                target_edge_index[0].append(edge[0] + offset)
                target_edge_index[1].append(edge[2] + offset)
                target_edge_feature.append([edge[1]])

            target_x += d["intra_interval_x"]
            target_node_ids += [index for i in range(numOfNodes)]
            target_calling_mask += d["calling_mask"]
            target_interval_node_ids += [0]
            if entryNode[0] == index:
                entry_node = entryNode[1] + offset
            offset += numOfNodes
        target_interval_offset = len(target)

        offset = 0
        graph_id = 0
        interval_offset = 0
        calleeNum = len(callees)
        for intervals in callees:
            numofIntervals = len(intervals)
            for index in range(numofIntervals):
                d = intervals[index]
                numOfNodes = len(d["intra_interval_x"])
                for edge in d["graph"]:
                    callee_edge_index[0].append(edge[0] + offset)
                    callee_edge_index[1].append(edge[2] + offset)
                    callee_edge_feature.append([edge[1]])

                callee_x += d["intra_interval_x"]
                callee_node_ids += [index + interval_offset for i in range(numOfNodes)]
                callee_interval_node_ids += [graph_id]
                offset += numOfNodes

            interval_offset += numofIntervals
            graph_id += 1
        callee_interval_offset = interval_offset

        caller_x = torch.tensor(caller_x).float()
        target_x = torch.tensor(target_x).float()
        callee_x = torch.tensor(callee_x).float()
        caller_edge_index = torch.tensor(caller_edge_index).long()
        target_edge_index = torch.tensor(target_edge_index).long()
        callee_edge_index = torch.tensor(callee_edge_index).long()
        caller_edge_feature = torch.tensor(caller_edge_feature).float()
        target_edge_feature = torch.tensor(target_edge_feature).float()
        callee_edge_feature = torch.tensor(callee_edge_feature).float()
        caller_node_ids = torch.tensor(caller_node_ids).long()
        target_node_ids = torch.tensor(target_node_ids).long()
        callee_node_ids = torch.tensor(callee_node_ids).long()
        caller_interval_node_ids = torch.tensor(caller_interval_node_ids).long()
        target_interval_node_ids = torch.tensor(target_interval_node_ids).long()
        callee_interval_node_ids = torch.tensor(callee_interval_node_ids).long()
        caller_offset = torch.tensor(caller_offset).long()
        target_offset = torch.tensor(target_offset).long()
        entry_node = torch.tensor(entry_node).long()
        caller_empty_mask = torch.tensor(caller_empty_mask).bool()

        data = EGINNData(intra_interval_x=[caller_x, target_x, callee_x], intra_interval_edge_index=[caller_edge_index, target_edge_index, callee_edge_index], intra_interval_edge_attr=[caller_edge_feature, target_edge_feature, callee_edge_feature],
                         intra_interval_node_ids=[caller_node_ids, target_node_ids, callee_node_ids], inter_interval_node_ids=[caller_interval_node_ids, target_interval_node_ids, callee_interval_node_ids],
                         offset=[caller_offset, target_offset], interval_offset=[caller_interval_offset, target_interval_offset, callee_interval_offset] ,entry_node=entry_node)
        data.caller_num = callerNum
        data.callee_num = calleeNum
        data.caller_calling_mask = caller_calling_mask
        data.target_calling_mask = target_calling_mask
        data.caller_empty_mask = caller_empty_mask

        return data

    class ROLE(Enum):
        CALLER = 1
        TARGET = 2
        CALLEE = 3

    def process_intervals(self, intervals, role : ROLE):
        intervalGraph = []

        for key in intervals:
            if key == "targets":
                intervalTarget = intervals[key]
                continue
            elif key == "insideinterval":
                continue
            elif key == "numOfNode":
                numOfNode = intervals[key]
                continue
            elif key == "graph":
                # (intervalAdjLists, intervalNIEPT) = self.__graph_to_adjacency_lists(intervals[key])
                continue
            elif key == "bugPos":
                intervalBugPos = intervals[key]
                continue
            elif key == "fileHash":
                intervalFileHash = intervals[key]
                continue
            elif key == "funName":
                intervalFunName = intervals[key]
                continue
            elif key == "entryNode":
                intervalEntryNode = intervals[key]
                continue
            d = intervals[key]
            # totalNumOfNodes += len(d["node_features"])

            assert len(d["node_features"]) != 0
            graphIndex = int(key)

            x = []
            for nf in d["node_features"]:
                x.append(nf)

            callingMask = []
            if role == self.ROLE.CALLER:
                for m in d["calling_mask"]:
                    if m == False:
                        callingMask.append([])
                    else:
                        callingMask.append([0])
            if role == self.ROLE.TARGET:
                CM = []
                for cm in d["calling_mask"]:
                    CM.append(cm)
                if len(CM) > 0:
                    CM = np.column_stack(CM)
                if len(CM) > 0:
                    for cm in CM:
                        mask = []
                        for i in range(CM.shape[1]):
                            if cm[i] == True:
                                mask.append(i)
                        callingMask.append(mask)
                else:
                    for i in range(len(d["node_features"])):
                        callingMask.append([])

            intervalGraph.append({"intra_interval_x": x,
                                  "graph": d["graph"],
                                  "calling_mask": callingMask,
                                  "graphIndex": graphIndex})

        intervalGraph = sorted(intervalGraph, key=lambda k: k['graphIndex'])
        return intervalGraph, intervalTarget[0], intervals["graph"], intervals["numOfNode"], intervalEntryNode

    def process(self):
        # Read data into huge `Data` list.
        jsonData = []
        with open(self.urlOrPath, 'r') as f:
            jsonData += json.load(f)
        data_list = []
        totalNum = len(jsonData)
        trainNum = int(totalNum*7/10)
        valNum = int(totalNum*3/20)
        intervalGraphList = []
        for index in range(totalNum):
            intervals = jsonData[index]

            intervalGraph = []
            callers, callees = [], []
            caller_graph, target_graph, callee_graph = [], [], []
            caller_num, target_num, callee_num = [], [], []
            intervalProjName = intervals["projName"]

            for caller_intervals in intervals["callers"]:
                d, t, g, n, _ = self.process_intervals(caller_intervals, self.ROLE.CALLER)
                callers.append(d)
                caller_graph.append(g)
                caller_num.append(n)

            target, target_target, g, n, entry_node = self.process_intervals(intervals["target"], self.ROLE.TARGET)
            target_graph.append(g)
            target_num.append(n)

            for callee_intervals in intervals["callees"]:
                d, t, g, n, _ = self.process_intervals(callee_intervals, self.ROLE.CALLEE)
                callees.append(d)
                callee_graph.append(g)
                callee_num.append(n)

            intervalData = self.processIntervalGraph(callers, target, callees, entry_node)
            # set intra_interval_node_ids


            # nodes in the same graph have the same id
            node_ID = []

            caller_edge_index, caller_edge_feature, caller_num_nodes = self.graph_to_COO(caller_graph, caller_num)
            target_edge_index, target_edge_feature, target_num_nodes = self.graph_to_COO(target_graph, target_num)
            callee_edge_index, callee_edge_feature, callee_num_nodes = self.graph_to_COO(callee_graph, callee_num)
            y = torch.tensor(target_target).long()
            intervalData.caller_inter_interval_edge_index = caller_edge_index
            intervalData.caller_inter_interval_edge_attr = caller_edge_feature
            intervalData.caller_num_nodes = caller_num_nodes
            intervalData.target_inter_interval_edge_index = target_edge_index
            intervalData.target_inter_interval_edge_attr = target_edge_feature
            intervalData.target_num_nodes = target_num_nodes
            intervalData.callee_inter_interval_edge_index = callee_edge_index
            intervalData.callee_inter_interval_edge_attr = callee_edge_feature
            intervalData.callee_num_nodes = callee_num_nodes
            intervalData.y=y
            data_list.append(intervalData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return
        data.train_mask = torch.full_like(data.y, False, dtype=bool)
        data.train_mask[[i for i in range(trainNum)]] = True
        data.val_mask = torch.full_like(data.y, False, dtype=bool)
        data.val_mask[[i for i in range(trainNum, trainNum+valNum)]] = True
        data.test_mask = torch.full_like(data.y, False, dtype=bool)
        data.test_mask[[i for i in range(trainNum+valNum, totalNum)]] = True
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
