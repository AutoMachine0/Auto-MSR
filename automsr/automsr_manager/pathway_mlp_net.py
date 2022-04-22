import torch
from torch_geometric.nn.inits import glorot
from automsr.gnn_model.stack_gcn_encoder.gcn_encoder import GcnEncoder

class PathWayMlpNet(torch.nn.Module):

    def __init__(self,
                 architecture,
                 data,
                 dropout=0.0):

        super(PathWayMlpNet, self).__init__()

        self.data = data
        self.embedding_dict_dim = self.data.embedding_dict_dim
        self.dropout = dropout
        self.molecule_embedding = torch.nn.Parameter(torch.Tensor(self.embedding_dict_dim,
                                                                  self.data.node_dim))
        self.reset_parameters()

        self.gnn_architecture = architecture

        self.output_dim = int(self.gnn_architecture[8])

        self.gnn = GcnEncoder(architecture,
                              self.data.node_dim,
                              dropout=self.dropout)

        self.linear_transformation = torch.nn.Linear(self.output_dim + data.global_feature_dim, data.label_num)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, batch, global_f, edge_index):

        x_feature = self.molecule_embedding[x]

        graph_embedding = self.gnn(x_feature, edge_index, batch)

        combination_embedding = torch.cat((graph_embedding, global_f), -1)

        output = self.linear_transformation(combination_embedding)

        return output

    def reset_parameters(self):

        glorot(self.molecule_embedding)