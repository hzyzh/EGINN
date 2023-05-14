import torch
import shutil, json, os
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data


class PDGDataset(InMemoryDataset):
    urlOrPath = "./input.json"
    savedFileName = "train.pt"
    def __init__(self, inputJsonPath, root, savedFileName, transform=None, pre_transform=None, pre_filter=None):
        self.urlOrPath = inputJsonPath
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
        shutil.copyfile(self.urlOrPath, os.path.join(self.raw_dir, self.urlOrPath))

    def graph_to_COO(self, graph):
        row = []
        col = []
        edge_feature = []
        for edge in graph:
            row.append(edge[0])
            col.append(edge[2])
            edge_feature.append([edge[1]])
        edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
        edge_feature = torch.tensor(edge_feature)
        return edge_index, edge_feature


    def process(self):
        # Read data into huge `Data` list.
        jsonData = []
        with open(self.raw_paths[0], 'r') as f:
            jsonData = json.load(f)
        data_list = []
        totalNum = len(jsonData)
        trainNum = int(totalNum*7/10)
        valNum = int(totalNum*3/20)
        for index in range(totalNum):
            #TODO: not consider batch
            d = jsonData[index]
            x = []
            # nodes in the same graph have the same id
            node_ID = []
            edge_index, edge_feature = self.graph_to_COO(d["graph"])
            for nf in d["node_features"]:
                x.append(nf)
                node_ID.append(index)
            x = torch.tensor(x).float()
            y = torch.tensor(d["targets"][0]).long()
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_feature, y=y)
            data.nodeIDs = torch.tensor(node_ID)
            data_list.append(data)

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
        data, _ = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
