from automsr.automsr_manager.pathway_gnn_manager import PathWayGnnManager

class Estimation(object):

    def __init__(self,
                 gnn_architecture,
                 data,
                 gnn_parameter):

        if not isinstance(gnn_architecture, list):
            raise Exception("gnn_architecture Class Wrong, require list Class ", "but input Class:",
                            type(gnn_architecture))

        if not isinstance(gnn_parameter, dict):
            raise Exception("gnn_parameter Class Wrong, require dict Class ", "but input Class:",
                            type(gnn_parameter))

        self.gnn_architecture = gnn_architecture
        self.data = data
        self.gnn_parameter = gnn_parameter

    def get_performance(self):

        if len(self.gnn_architecture) == 0 or not isinstance(self.gnn_architecture, list):
            raise Exception("wrong gnn_architecture:", self.gnn_architecture)
        if not isinstance(self.gnn_parameter, dict):
            raise Exception("gnn_parameter", self.gnn_parameter)

        # GNNs default training parameter
        drop_out = 0.60
        learning_rate = 0.005
        learning_rate_decay = 0.0005
        train_epoch = 300
        model_select = "min_loss"
        one_layer_component_num = 5
        early_stop = True
        early_stop_mode = "val_loss"
        early_stop_patience = 10

        if "drop_out" in self.gnn_parameter:
            drop_out = self.gnn_parameter["drop_out"]
        if "learning_rate" in self.gnn_parameter:
            learning_rate = self.gnn_parameter["learning_rate"]
        if "weight_decay" in self.gnn_parameter:
            learning_rate_decay = self.gnn_parameter["learning_rate_decay"]
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

        # GNN default training mode
        es_mode = "transductive"

        if es_mode == "transductive":

            model = PathWayGnnManager(drop_out,
                                      learning_rate,
                                      learning_rate_decay,
                                      train_epoch,
                                      model_select,
                                      one_layer_component_num,
                                      early_stop,
                                      early_stop_mode,
                                      early_stop_patience)

            model.build_model(self.gnn_architecture, self.data)
            auc_val, precision_val, recall_val, f1score_val = model.train()

        return recall_val
