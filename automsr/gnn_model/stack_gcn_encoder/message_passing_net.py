import torch
import copy
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from automsr.dynamic_configuration import attention_getter
from torch_geometric.utils import remove_self_loops, add_self_loops
from automsr.gnn_model.stack_gcn_encoder.message_passing import MessagePassing

class MessagePassingNet(MessagePassing):

    def __init__(self,
                 input_dim,
                 output_dim,
                 heads=1,
                 concat=True,
                 dropout=0,
                 bias=True,
                 att_type="gcn",
                 agg_type="sum"):

        super(MessagePassingNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.att_type = att_type
        self.agg_type = agg_type
        self.bias = bias

        self.weight = Parameter(torch.Tensor(self.heads,
                                             self.input_dim,
                                             self.output_dim))
        glorot(self.weight)

        if self.bias and concat:
            self.bias = Parameter(torch.Tensor(self.heads * self.output_dim))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(self.output_dim))
        else:
            self.bias = None

        if self.bias is not None:
            zeros(self.bias)

        self.attention_dict = attention_getter(self.heads, self.output_dim)

    def forward(self, x, edge_index):

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        node_nums = x.shape[0]

        for weight_index in range(self.weight.shape[0]):

            if weight_index == 0:
                x_ = torch.mm(x, self.weight[weight_index])
                edge_index_ = copy.deepcopy(edge_index)
            else:
                edge_index = edge_index + node_nums
                x_ = torch.cat([x_, torch.mm(x, self.weight[weight_index])], dim=0)
                edge_index_ = torch.cat([edge_index_, edge_index], dim=-1)

        return self.propagate(edge_index_, x=x_, num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_index, num_nodes):

        self.edge_index = edge_index

        attention_function = self.attention_dict[self.att_type]

        attention_coefficient = attention_function.function(x_i,
                                                            x_j,
                                                            edge_index,
                                                            num_nodes)

        self.source_node_representation_with_coefficent = attention_coefficient * x_j

        if self.training and self.dropout > 0:

            self.source_node_representation_with_coefficent = F.dropout(self.source_node_representation_with_coefficent,
                                                                        p=self.dropout,
                                                                        training=True)

        return self.source_node_representation_with_coefficent

    def update(self, aggr_out):

        node_representation = aggr_out

        node_representation = self.node_representation_transformer(node_representation)

        return node_representation


    def node_representation_transformer(self, node_representation_):

        node_representation_ = node_representation_.view(self.heads,
                                                         int(node_representation_.shape[0]/self.heads),
                                                         self.output_dim)

        if self.concat is True:
            for index in range(self.heads):
                if index == 0:
                    node_representation = node_representation_[index]
                else:
                    node_representation = torch.cat([node_representation,
                                                    node_representation_[index]],
                                                    dim=1)
        else:
            for index in range(self.heads):
                if index == 0:
                    node_representation = node_representation_[index]
                else:
                    node_representation = node_representation + node_representation_[index]

            node_representation = node_representation / self.heads

        if self.bias is not None:
            node_representation = node_representation + self.bias

        return node_representation

if __name__=="__main__":

    pass