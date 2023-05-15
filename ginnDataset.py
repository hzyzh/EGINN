import torch
import shutil, json, os
from glob import glob
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from dataset import PDGDataset



class GINNData(Data):
    def __init__(self, intra_interval_x=None, intra_interval_edge_index=None, intra_interval_edge_attr=None, intra_interval_node_ids=None):
        super().__init__()
        self.intra_interval_x = intra_interval_x
        self.intra_interval_edge_index = intra_interval_edge_index
        self.intra_interval_edge_attr = intra_interval_edge_attr
        self.intra_interval_node_ids = intra_interval_node_ids
        self.inter_interval_edge_attr = None
        self.inter_interval_edge_index = None
        self.y = None
        self.num_nodes = None

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'intra_interval_node_ids':
            return self.intra_interval_node_ids[-1] + 1
        if key == 'intra_interval_edge_index':
            return self.intra_interval_x.size(0)
        if key == 'inter_interval_edge_index':
            return self.num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)

class GINNDataset(InMemoryDataset):
    urlOrPath = "./input.json"
    savedFileName = "train.pt"
    def __init__(self, inputJsonPath, root, savedFileName, transform=None, pre_transform=None, pre_filter=None):
        # self.urlOrPath = inputJsonPath
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
        '''
        shutil.copyfile(self.urlOrPath, os.path.join(self.raw_dir, self.urlOrPath))
        '''
        jsonData = []
        for fileName in self.inputJsonPath:
            with open(fileName, 'r') as f:
                jsonData += json.load(f)
        self.urlOrPath = os.path.join(self.raw_dir, "GINN-"+self.savedFileName+".json")

        with open(self.urlOrPath, "w") as outfile:
            json.dump(jsonData, outfile)

    def graph_to_COO(self, graph, offset=0):
        row = []
        col = []
        edge_feature = []
        for edge in graph:
            row.append(edge[0]+offset)
            col.append(edge[2]+offset)
            edge_feature.append([edge[1]])
        edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
        edge_feature = torch.tensor(edge_feature)
        return edge_index, edge_feature

    def processIntervalGraph(self, intervalGraph):
        x = []
        edge_index = [[], []]
        edge_feature = []
        offset = 0
        node_ids = []

        for index in range(len(intervalGraph)):
            d = intervalGraph[index]
            numOfNodes = len(d["intra_interval_x"])
            # process edge_index is complex, should add offset to index
            for edge in d["graph"]:
                edge_index[0].append(edge[0]+offset)
                edge_index[1].append(edge[2]+offset)
                edge_feature.append([edge[1]])

            x += d["intra_interval_x"]
            offset += numOfNodes
            node_ids += [index for i in range(numOfNodes)]


        x = torch.tensor(x).float()
        edge_index = torch.tensor(edge_index)
        edge_feature = torch.tensor(edge_feature)
        node_ids = torch.tensor(node_ids)
        data = GINNData(intra_interval_x=x, intra_interval_edge_index=edge_index, intra_interval_edge_attr=edge_feature, intra_interval_node_ids=node_ids)
        return data

    def process(self):
        # Read data into huge `Data` list.
        jsonData = []
        with open(self.raw_paths[0], 'r') as f:
            jsonData += json.load(f)
        data_list = []
        totalNum = len(jsonData)
        trainNum = int(totalNum*7/10)
        valNum = int(totalNum*3/20)
        intervalGraphList = []
        for index in range(totalNum):
            intervals = jsonData[index]

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
                    #(intervalAdjLists, intervalNIEPT) = self.__graph_to_adjacency_lists(intervals[key])
                    continue
                elif key == "bugPos":
                    intervalBugPos = intervals[key]
                    continue
                elif key == "fileHash":
                    intervalFileHash = intervals[key]
                    continue
                elif key == "projName":
                    intervalProjName = intervals[key]
                    continue
                elif key == "funName":
                    intervalFunName = intervals[key]
                    continue
                d = intervals[key]
                #totalNumOfNodes += len(d["node_features"])

                assert len(d["node_features"]) != 0
                graphIndex = int(key)

                x = []
                for nf in d["node_features"]:
                    x.append(nf)

                intervalGraph.append({"intra_interval_x": x,
                    "graph": d["graph"],
                    "graphIndex": graphIndex})

            intervalGraph = sorted(intervalGraph, key=lambda k: k['graphIndex'])

            intervalData = self.processIntervalGraph(intervalGraph)
            # set intra_interval_node_ids


            # nodes in the same graph have the same id
            node_ID = []
            edge_index, edge_feature = self.graph_to_COO(intervals["graph"])
            y = torch.tensor(intervalTarget[0]).long()
            intervalData.inter_interval_edge_index=edge_index
            intervalData.inter_interval_edge_attr=edge_feature
            intervalData.y=y
            intervalData.num_nodes=intervals["numOfNode"]
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
