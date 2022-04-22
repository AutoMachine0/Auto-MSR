from hyperopt import fmin, tpe
from automsr.hp_search.hp_search_space import HP_SEARCH_SPACE, HP_SEARCH_SPACE_Mapping
from pathway_utils import PathWay
from automsr.automsr_manager.pathway_gnn_manager_hp import PathWayGnnManager

class HpSearchObj():

    def __init__(self, gnn_architecture, data, gnn_parameter):
        self.gnn_architecture = gnn_architecture
        self.data = data
        self.gnn_parameter = gnn_parameter

    def tuining_obj(self, hp_space):

        if len(self.gnn_architecture) == 0 or not isinstance(self.gnn_architecture, list):
            raise Exception("wrong gnn_architecture:", self.gnn_architecture)
        if not isinstance(self.gnn_parameter, dict):
            raise Exception("gnn_parameter", self.gnn_parameter)

        # GNNs default training parameter
        drop_out = 0.0
        learning_rate = 0.001
        learning_rate_decay = 0.0005
        mini_batch = 20
        embedding_dim = 70
        train_epoch = 300
        model_select = "min_loss"
        one_layer_component_num = 5
        early_stop = True
        early_stop_mode = "val_loss"
        early_stop_patience = 10

        if "learning_rate" in hp_space:
            learning_rate = hp_space["learning_rate"]
        if "learning_rate_decay" in hp_space:
            learning_rate_decay = hp_space["learning_rate_decay"]
        if "mini_batch" in hp_space:
            mini_batch = hp_space["mini_batch"]
        if "embedding_dim" in hp_space:
            embedding_dim = hp_space["embedding_dim"]

        print(32 * "#" + " 本轮训练超参 " + 32 * "#")
        print("learning_rate: %f / learning_rate_decay: %f / mini_batch: %f / embedding_dim: %f" % (
        learning_rate, learning_rate_decay, mini_batch, embedding_dim))
        print(32 * "#" + " 本轮训练超参 " + 32 * "#")

        # train_epoch = 10 for unit test

        if "train_epoch" in self.gnn_parameter:
            train_epoch = self.gnn_parameter["train_epoch"]
        if "model_select" in self.gnn_parameter:
            model_select = self.gnn_parameter["model_select"]
        if "one_layer_component_num" in self.gnn_parameter:
            one_layer_component_num = self.gnn_parameter["one_layer_component_num"]
        if "early_stop" in self.gnn_parameter:
            early_stop = self.gnn_parameter["early_stop"]
        if "early_mode" in self.gnn_parameter:
            early_stop_mode = self.gnn_parameter["early_stop_mode"]
        if "early_num" in self.gnn_parameter:
            early_stop_patience = self.gnn_parameter["early_stop_patience"]

        model = PathWayGnnManager(drop_out,
                                  learning_rate,
                                  learning_rate_decay,
                                  train_epoch,
                                  model_select,
                                  one_layer_component_num,
                                  early_stop,
                                  early_stop_mode,
                                  early_stop_patience)

        data = PathWay(0.8, 0.1, mini_batch, embedding_dim)

        model.build_model(self.gnn_architecture, data)

        auc_val, precision_mean, recall_val, f1score = model.train()

        return -precision_mean


    def hp_tuning(self, search_epoch, search_algorithm):
        print("target_gnn_architecture:", self.gnn_architecture)

        best_hp = fmin(fn=self.tuining_obj,
                       space=HP_SEARCH_SPACE,
                       algo=search_algorithm,
                       max_evals=search_epoch)

        learning_rate_index = best_hp["learning_rate"]
        learning_rate_decay_index = best_hp["learning_rate_decay"]
        mini_batch_index = best_hp["mini_batch"]
        embedding_dim_index = best_hp["embedding_dim"]

        learning_rate = HP_SEARCH_SPACE_Mapping["learning_rate"][learning_rate_index]
        learning_rate_decay = HP_SEARCH_SPACE_Mapping["learning_rate_decay"][learning_rate_decay_index]
        mini_batch = HP_SEARCH_SPACE_Mapping["mini_batch"][mini_batch_index]
        embedding_dim = HP_SEARCH_SPACE_Mapping["embedding_dim"][embedding_dim_index]

        print("the optimal learning_rate: %f / learning_rate_decay: %f / mini_batch: %f / embedding_dim: %f" % (
            learning_rate, learning_rate_decay, mini_batch, embedding_dim))

        return learning_rate, learning_rate_decay, mini_batch, embedding_dim

if __name__=="__main__":
    pass