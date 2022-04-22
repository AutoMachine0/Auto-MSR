import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm
from automsr.dynamic_configuration import activation_getter
from torch_scatter import scatter_mean, scatter_add, scatter_max
from automsr.gnn_model.stack_gcn_encoder.message_passing_net import MessagePassingNet

class GcnEncoder(torch.nn.Module):

    def __init__(self,
                 architecture,
                 original_feature_num,
                 dropout=0.0,
                 bias=True):

        super(GcnEncoder, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.architecture = architecture
        self.original_feateure_num = original_feature_num
        self.bias = bias
        self.layer_num = 2
        self.dropout = dropout
        self.activation_dict = activation_getter()

        self.activation_operators = []
        self.global_pooling_operators = []
        self.conv_layers = torch.nn.ModuleList()
        self.pooling_layers = torch.nn.ModuleList()
        self.normalization_layers = torch.nn.ModuleList()
        hidden_dimension_num = None
        multi_heads_num = None

        for layer in range(self.layer_num):

            if layer == 0:
                concat = True
                input_dim = self.original_feateure_num
            else:
                concat = False
                input_dim = hidden_dimension_num * multi_heads_num

            attention_type = self.architecture[layer * 5 + 0]
            aggregator_type = self.architecture[layer * 5 + 1]
            multi_heads_num = int(self.architecture[layer * 5 + 2])
            hidden_dimension_num = int(self.architecture[layer * 5 + 3])
            activation_type = self.architecture[layer * 5 + 4]

            if layer == 0:
                graphnorm = GraphNorm(hidden_dimension_num * multi_heads_num)
            else:
                graphnorm = GraphNorm(hidden_dimension_num)

            self.conv_layers.append(MessagePassingNet(input_dim,
                                                      hidden_dimension_num,
                                                      multi_heads_num,
                                                      concat,
                                                      dropout=self.dropout,
                                                      bias=self.bias,
                                                      att_type=attention_type,
                                                      agg_type=aggregator_type))

            self.activation_operators.append(self.activation_dict[activation_type])
            self.normalization_layers.append(graphnorm)

    def forward(self, x, edge_index_all, batch):

        output = x
        for activation, conv_layer, graph_norm in zip(self.activation_operators,
                                                      self.conv_layers,
                                                      self.normalization_layers):
            # dropout for node embedding matrix
            output = F.dropout(output, p=self.dropout, training=self.training)
            # convolution for node embedding matrix
            output = conv_layer(output, edge_index_all)
            # GraphNorm
            output = graph_norm(output, batch)
            # activation for node embedding matrix
            output = activation(output)

        # last layer readout
        output = self.global_pooling_operator("g_sum", output, batch)
        return output

    def global_pooling_operator(self, readout_name, input, batch):

        readout_output = 0

        if readout_name == "g_sum":
            readout_output = self.global_sum_pooling(input, batch)
        elif readout_name == "g_mean":
            readout_output = self.global_mean_pooling(input, batch)
        elif readout_name == "g_max":
            readout_output = self.global_mean_pooling(input, batch)

        return readout_output

    def global_mean_pooling(self,
                            batch_node_embedding_matrix,
                            index):

        graph_embedding = scatter_mean(batch_node_embedding_matrix, index, dim=0)
        return graph_embedding

    def global_sum_pooling(self,
                           batch_node_embedding_matrix,
                           index):

        graph_embedding = scatter_add(batch_node_embedding_matrix, index, dim=0)
        return graph_embedding

    def global_max_pooling(self,
                           batch_node_embedding_matrix,
                           index):

        graph_embedding, _ = scatter_max(batch_node_embedding_matrix, index, dim=0)
        return graph_embedding

if __name__=="__main__":

    pass
