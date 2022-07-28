import numpy as np
import os
import torch

class PathWay():

    def __init__(self, train_ratio, val_ratio, batch_size, node_dim):

        path = os.path.split(os.path.realpath(__file__))[0]+"/pathway_data/"
        adj = "adjacencies.npy"
        maccs = "maccs.npy"
        label = "properties.npy"
        node_dict_number = "molecules.npy"
        adjacencies = "adjacencies.npy"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.adj = np.load(path+adj, allow_pickle=True)
        self.maccs = np.load(path+maccs, allow_pickle=True)
        self.label = np.load(path+label, allow_pickle=True)

        self.graph_data = np.load(path+node_dict_number, allow_pickle=True)
        self.adjacencies = np.load(path + adjacencies, allow_pickle=True)

        # fix
        data_index = []
        file_name = "data_index.txt"
        with open(path + "/" + file_name, "r") as f:
            for line in f.readlines():
                line = eval(line)
                data_index.append(line)

        self.train_index = data_index[0]
        self.val_index = data_index[1]
        self.test_index = data_index[2]

        self.train_data = []
        self.train_data_maccs = []
        self.train_adj = []
        self.train_data_label = []

        self.val_data = []
        self.val_data_maccs = []
        self.val_adj = []
        self.val_data_label = []

        self.test_data = []
        self.test_data_maccs = []
        self.test_adj = []
        self.test_data_label = []

        for index in self.train_index:
            self.train_data.append(self.graph_data[index])
            self.train_data_maccs.append(self.maccs[index])
            self.train_adj.append(self.adjacencies[index])
            self.train_data_label.append(self.label[index])

        for index in self.val_index:
            self.val_data.append(self.graph_data[index])
            self.val_data_maccs.append(self.maccs[index])
            self.val_adj.append(self.adjacencies[index])
            self.val_data_label.append(self.label[index])

        for index in self.test_index:
            self.test_data.append(self.graph_data[index])
            self.test_data_maccs.append(self.maccs[index])
            self.test_adj.append(self.adjacencies[index])
            self.test_data_label.append(self.label[index])

        self.embedding_dict_dim = 3617
        self.node_dim = node_dim
        self.global_feature_dim = len(self.train_data_maccs[0])
        self.label_num = len(self.train_data_label[0][0])
        self.combination_dim = node_dim + self.global_feature_dim

        # adj to list
        self.train_edge_index_list = self.edge_index_operator(self.train_adj)
        self.val_edge_index_list = self.edge_index_operator(self.val_adj)
        self.test_edge_index_list = self.edge_index_operator(self.test_adj)

        # list to torch
        self.train_data_maccs = torch.tensor(self.train_data_maccs, dtype=torch.float32).to(self.device)
        self.train_data_label = torch.squeeze(torch.tensor(self.train_data_label, dtype=torch.float32)).to(self.device)

        self.val_data_maccs = torch.tensor(self.val_data_maccs, dtype=torch.float32).to(self.device)
        self.val_data_label = torch.squeeze(torch.tensor(self.val_data_label, dtype=torch.float32)).to(self.device)

        self.test_data_maccs = torch.tensor(self.test_data_maccs, dtype=torch.float32).to(self.device)
        self.test_data_label = torch.squeeze(torch.tensor(self.test_data_label, dtype=torch.float32)).to(self.device)

        self.val_data, \
        self.val_data_len, \
        self.val_edge_index_list, \
        self.val_batch = self.val_test_data_merge(self.val_data,
                                                  self.val_edge_index_list)

        self.test_data, \
        self.test_data_len, \
        self.test_edge_index_list, \
        self.test_batch = self.val_test_data_merge(self.test_data,
                                                   self.test_edge_index_list)

        # training data batch_size operator
        if batch_size != 0:
            self.train_data_batch_list, \
            self.train_data_len_batch_list, \
            self.train_maccs_batch_list, \
            self.train_label_batch_list, \
            self.train_edge_index_batch_list, self.train_batch_list = self.batch_szie_operator(batch_size,
                                                                      self.train_data,
                                                                      self.train_data_maccs,
                                                                      self.train_data_label,
                                                                      self.train_edge_index_list)

        self.val_data = self.val_data
        self.val_data_len = self.val_data_len
        self.val_edge_index_list = self.val_edge_index_list.to(self.device)
        self.test_data = self.test_data
        self.test_data_len = self.test_data_len
        self.test_edge_index_list = self.test_edge_index_list.to(self.device)
        self.train_data_batch_list = self.train_data_batch_list
        self.train_data_len_batch_list = self.train_data_len_batch_list
        self.train_maccs_batch_list = self.train_maccs_batch_list
        self.data_name = "pathway"

    def val_test_data_merge(self, data, edge_index_list):

        merge_data = []
        merge_data_len = []
        graph_id = 0
        batch = []
        static = 0

        for sub_data in data:
            merge_data = merge_data + list(sub_data)
            merge_data_len.append(len(sub_data))
            static += len(sub_data)
            if batch == []:
                batch = [graph_id for i in range(len(sub_data))]
            else:
                batch.extend([graph_id for i in range(len(sub_data))])
            graph_id += 1

        for index in range(len(edge_index_list)):
            if index == 0:
                source_list = edge_index_list[index][0].tolist()
                target_list = edge_index_list[index][1].tolist()
                max_number = max(source_list) + 1
            else:
                source_list_temp = edge_index_list[index][0].tolist()
                target_list_temp = edge_index_list[index][1].tolist()
                max_number_temp = max(source_list_temp) + 1
                source_list_temp = [i + max_number for i in source_list_temp]
                target_list_temp = [i + max_number for i in target_list_temp]
                source_list = source_list + source_list_temp
                target_list = target_list + target_list_temp
                max_number = max_number + max_number_temp

        merge_edge_list = torch.tensor([source_list, target_list], dtype=torch.int64)

        batch = torch.tensor(batch, dtype=torch.int64).to(self.device)

        return merge_data, merge_data_len, merge_edge_list, batch

    def edge_index_operator(self, adj):
        edge_index_list = []
        adj_p = []
        for index in range(len(adj)):
            adj_temp = adj[index]
            source_list = []
            target_list = []
            for j in range(adj_temp.shape[0]):
                for i in range(adj_temp.shape[0]):
                    if (adj_temp[i][j] != 0) and (i != j):
                        source_list.append(j)
                        target_list.append(i)
            if source_list == [] or target_list == []:
                adj_p.append(adj_temp)
                print(adj_temp)
            edge_index = [source_list, target_list]
            edge_index = torch.tensor(edge_index, dtype=torch.int64)
            edge_index_list.append(edge_index)
        return edge_index_list

    def batch_szie_operator(self, batch_size, data, data_maccs, data_label, edge_index_list):

        data_batch_list = []
        data_batch_len_list = []
        maccs_batch_list = []
        label_batch_list = []
        edge_index_batch_list = []
        start_index = 0
        end_index = batch_size
        batch_list = []

        for i in range(int(len(data) / batch_size)):

            batch = []
            graph_id = 0

            one_batch_data = []
            one_batch_data_len = []

            # one batch dataset merge
            for sub_data in data[start_index:end_index]:
                one_batch_data = one_batch_data + list(sub_data)
                one_batch_data_len.append(len(sub_data))

                if batch == []:
                    batch = [graph_id for i in range(len(sub_data))]
                else:
                    batch.extend([graph_id for i in range(len(sub_data))])
                graph_id += 1

            one_batch_maccs = []

            for sub_maccs in data_maccs[start_index:end_index]:
                if isinstance(one_batch_maccs, list):
                    one_batch_maccs = sub_maccs
                else:
                    one_batch_maccs = torch.cat(
                        (one_batch_maccs.view(-1, len(sub_maccs)), sub_maccs.view(-1, len(sub_maccs))), 0)

            one_batch_label = []

            for sub_label in data_label[start_index:end_index]:
                if isinstance(one_batch_label, list):
                    one_batch_label = sub_label
                else:
                    one_batch_label = torch.cat(
                        (one_batch_label.view(-1, len(sub_label)), sub_label.view(-1, len(sub_label))), 0)

            #edge_index_batch_list.append(edge_index_list[start_index:end_index])

            for index in range(len(edge_index_list[start_index:end_index])):
                if index == 0:
                    source_list = edge_index_list[start_index:end_index][index][0].tolist()
                    target_list = edge_index_list[start_index:end_index][index][1].tolist()
                    max_number = max(source_list) + 1
                else:
                    source_list_temp = edge_index_list[start_index:end_index][index][0].tolist()
                    target_list_temp = edge_index_list[start_index:end_index][index][1].tolist()
                    max_number_temp = max(source_list_temp) + 1
                    source_list_temp = [i + max_number for i in source_list_temp]
                    target_list_temp = [i + max_number for i in target_list_temp]
                    source_list = source_list + source_list_temp
                    target_list = target_list + target_list_temp
                    max_number = max_number + max_number_temp

            one_batch_edge_index = torch.tensor([source_list, target_list], dtype=torch.int64).to(self.device)

            start_index += batch_size
            end_index += batch_size
            data_batch_list.append(one_batch_data)
            data_batch_len_list.append(one_batch_data_len)
            maccs_batch_list.append(one_batch_maccs)
            label_batch_list.append(one_batch_label)
            edge_index_batch_list.append(one_batch_edge_index)
            batch = torch.tensor(batch, dtype=torch.int64).to(self.device)
            batch_list.append(batch)

        # one batch dataset merge
        if (len(data) % batch_size) != 0:

            one_batch_data = []
            one_batch_data_len = []

            batch = []
            graph_id = 0

            for sub_data in data[start_index:end_index]:
                one_batch_data = one_batch_data + list(sub_data)
                one_batch_data_len.append(len(sub_data))

                if batch == []:
                    batch = [graph_id for i in range(len(sub_data))]
                else:
                    batch.extend([graph_id for i in range(len(sub_data))])
                graph_id += 1

            one_batch_maccs = []

            for sub_maccs in data_maccs[start_index:end_index]:
                if isinstance(one_batch_maccs, list):
                    one_batch_maccs = sub_maccs
                else:
                    one_batch_maccs = torch.cat(
                        (one_batch_maccs.view(-1, len(sub_maccs)), sub_maccs.view(-1, len(sub_maccs))), 0)

            one_batch_label = []

            for sub_label in data_label[start_index:end_index]:
                if isinstance(one_batch_label, list):
                    one_batch_label = sub_label
                else:
                    one_batch_label = torch.cat(
                        (one_batch_label.view(-1, len(sub_label)), sub_label.view(-1, len(sub_label))), 0)

            # edge_index_batch_list.append(edge_index_list[start_index:end_index])
            for index in range(len(edge_index_list[start_index:end_index])):
                if index == 0:
                    source_list = edge_index_list[start_index:end_index][index][0].tolist()
                    target_list = edge_index_list[start_index:end_index][index][1].tolist()
                    max_number = max(source_list) + 1
                else:
                    source_list_temp = edge_index_list[start_index:end_index][index][0].tolist()
                    target_list_temp = edge_index_list[start_index:end_index][index][1].tolist()
                    max_number_temp = max(source_list_temp) + 1
                    source_list_temp = [i + max_number for i in source_list_temp]
                    target_list_temp = [i + max_number for i in target_list_temp]
                    source_list = source_list + source_list_temp
                    target_list = target_list + target_list_temp
                    max_number = max_number + max_number_temp

            one_batch_edge_index = torch.tensor([source_list, target_list], dtype=torch.int64).to(self.device)

            data_batch_list.append(one_batch_data)
            data_batch_len_list.append(one_batch_data_len)
            maccs_batch_list.append(one_batch_maccs)
            label_batch_list.append(one_batch_label)
            edge_index_batch_list.append(one_batch_edge_index)
            batch = torch.tensor(batch, dtype=torch.int64).to(self.device)
            batch_list.append(batch)

        return data_batch_list, data_batch_len_list, maccs_batch_list, label_batch_list, edge_index_batch_list, batch_list

def count(y, name):

    print(35 * "=", name, 35 * "=")

    column_num = y.shape[1]
    raw_num = y.shape[0]
    label = 0
    print("the total number of sample are: " + str(int(raw_num)))

    for c in range(column_num):
        label += 1
        class_num = int(y[:, c].sum())
        print("the label " + str(label) + " have:" + str(class_num))

    sample = 0
    for r in range(raw_num):
        class_num = y[r].sum()
        if class_num >= 2:
            sample += 1
    print("the number of sample that has two labels are: ", str(sample))





if __name__=="__main__":
    a = PathWay(0.8, 0.1, 10, 70)
    train_label = a.train_data_label.cpu().numpy()
    val_label = a.val_data_label.cpu().numpy()
    test_label = a.test_data_label.cpu().numpy()
    count(train_label, "train data")
    count(val_label, "val data")
    count(test_label, "test data")
