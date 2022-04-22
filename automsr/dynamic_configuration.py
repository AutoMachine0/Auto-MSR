import os
import torch.nn
import importlib

def search_algorithm_getter(search_algorithm_type,
                            data,
                            search_parameter,
                            gnn_parameter,
                            search_space):

    search_algorithm_class = "Search"
    search_algorithm_module = importlib.import_module("automsr.search_algorithm." +
                                                      search_algorithm_type +
                                                      "." +
                                                      "search_algorithm")

    search_algorithm_obj = getattr(search_algorithm_module,
                                   search_algorithm_class)

    search_algorithm = search_algorithm_obj(data,
                                            search_parameter,
                                            gnn_parameter,
                                            search_space)
    return search_algorithm

def optimizer_getter(opt_type,
                     gnn_model,
                     optimizer_parameter_dict):

    optimizer_class = "Optimizer"
    optimizer_module = importlib.import_module("automsr.model.optimizer_function" +
                                               "." +
                                               opt_type)
    optimizer_obj = getattr(optimizer_module, optimizer_class)
    optimizer_function = optimizer_obj(gnn_model,
                                       optimizer_parameter_dict).function()

    return optimizer_function


def loss_getter(loss_type):

    loss_class = "Loss"
    loss_module = importlib.import_module("automsr.model.loss_function" +
                                          "." +
                                          loss_type)
    loss_obj = getattr(loss_module, loss_class)
    loss_function = loss_obj()

    return loss_function


def evaluator_getter(evaluator_type):

    evaluator_class = "Evaluator"
    evaluator_module = importlib.import_module("automsr.model.evaluator_function" +
                                               "." +
                                               evaluator_type)
    evaluator_obj = getattr(evaluator_module, evaluator_class)
    evaluator_function = evaluator_obj()

    return evaluator_function


def downstream_task_model_getter(downstream_task_type,
                                 gnn_embedding_dim,
                                 graph_data):

    downstream_task_model_class = "DownstreamTask"
    downstream_task_model_module = importlib.import_module("automsr.model.downstream_task_model" +
                                                           "." +
                                                           downstream_task_type)
    downstream_task_model_obj = getattr(downstream_task_model_module, downstream_task_model_class)

    downstream_task = downstream_task_model_obj(gnn_embedding_dim, graph_data)

    return downstream_task

def attention_getter(heads,
                     output_dim):

    search_space_path = os.path.split(os.path.realpath(__file__))[0] + "/search_space/attention"
    attention_list = [attention for attention in os.listdir(search_space_path) if attention not in "__pycache__"
                      and attention not in "README.md"]
    attention_dict = torch.nn.ModuleDict()
    for attention in attention_list:
        attention_class = "Attention"
        attention_module = importlib.import_module("automsr.search_space.attention" +
                                                    "." + attention[:-3])
        attention_obj = getattr(attention_module, attention_class)
        attention_function = attention_obj(heads, output_dim)
        attention_dict[attention[:-3]] = attention_function

    return attention_dict

def aggregation_getter():

    search_space_path = os.path.split(os.path.realpath(__file__))[0]+ "/search_space/aggregation"
    aggregation_list = [aggregation for aggregation in os.listdir(search_space_path) if aggregation not in "__pycache__"
                        and aggregation not in "README.md"]
    aggregation_dict = {}
    for aggregation in aggregation_list:
        aggregation_class = "Aggregation"
        aggregation_module = importlib.import_module("automsr.search_space.aggregation" +
                                                     "." + aggregation[:-3])
        aggregation_obj = getattr(aggregation_module, aggregation_class)
        aggregation_function = aggregation_obj()
        aggregation_dict[aggregation[:-3]] = aggregation_function

    return aggregation_dict

def activation_getter():

    search_space_path = os.path.split(os.path.realpath(__file__))[0] + "/search_space/activation"
    activation_list = [activation for activation in os.listdir(search_space_path) if activation not in "__pycache__"
                       and activation not in "README.md"]
    activation_dict = {}
    for activation in activation_list:
        activation_class = "Activation"
        activation_module = importlib.import_module("automsr.search_space.activation" +
                                                    "." + activation[:-3])
        activation_obj = getattr(activation_module, activation_class)
        activation_function = activation_obj().function()
        activation_dict[activation[:-3]] = activation_function

    return activation_dict

if __name__=="__main__":
    pass