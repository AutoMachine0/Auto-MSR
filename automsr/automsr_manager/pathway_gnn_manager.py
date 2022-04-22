import time
import torch
import numpy
from automsr.automsr_manager import utils
from automsr.automsr_manager.pathway_mlp_net import PathWayMlpNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
import warnings
warnings.filterwarnings('ignore')

class PathWayGnnManager(object):

    def __init__(self,
                 drop_out=0.6,
                 learning_rate=0.005,
                 learning_rate_decay=0.0005,
                 train_epoch=300,
                 model_select="min_loss",
                 one_layer_component_num=5,
                 early_stop=True,
                 early_stop_mode="val_loss",
                 early_stop_patience=10):

        self.drop_out = drop_out
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.train_epoch = train_epoch
        self.model_select = model_select
        self.retrain_epoch = self.train_epoch
        self.one_layer_component_num = one_layer_component_num
        self.early_stop = early_stop
        self.early_stop_mode = early_stop_mode
        self.early_stop_patience = early_stop_patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self, architecture, data):

        print("\n", 35 * "=", "a GNNs estimation start", 35 * "=", "\n")
        print("build architecture:", architecture)

        self.architecture = architecture
        self.data = data
        self.model = PathWayMlpNet(self.architecture,
                                   self.data,
                                   dropout=self.drop_out)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.learning_rate_decay)

        self.loss_func = torch.nn.MultiLabelSoftMarginLoss()


    def train(self, test_epoch=None, test_mode=False):

        print("train architecture:", self.architecture)
        self.model.to(self.device)

        if test_mode:
            auc_val, precision_val, recall_val, f1score, class_information_list = self.run_model(self.model,
                                                                                                 self.optimizer,
                                                                                                 self.loss_func,
                                                                                                 self.data,
                                                                                                 self.train_epoch,
                                                                                                 test_epoch,
                                                                                                 test_mode=test_mode)
            return auc_val, precision_val, recall_val, f1score, class_information_list
        else:
            auc_val, precision_val, recall_val, f1score = self.run_model(self.model,
                                                                         self.optimizer,
                                                                         self.loss_func,
                                                                         self.data,
                                                                         self.train_epoch,
                                                                         test_epoch,
                                                                         test_mode=test_mode)

            return auc_val, precision_val, recall_val, f1score

    def evaluate(self, model, data, mode="test_data"):

        label_list, t_list = [], []

        if mode == "test_data":

            logits = model(data.test_data,
                           data.test_batch,
                           data.test_data_maccs,
                           data.test_edge_index_list)

            label_list, t_list = self.inference(logits, data.test_data_label)
            label_list = numpy.array(label_list)

        elif mode == "val_data":

            logits = model(data.val_data,
                           data.val_batch,
                           data.val_data_maccs,
                           data.val_edge_index_list)

            label_list, t_list = self.inference(logits, data.val_data_label)
            label_list = numpy.array(label_list)


        acc = self.Accuracy(t_list, label_list)
        precision = precision_score(t_list, label_list, average="samples")
        recall = recall_score(t_list, label_list, average="samples")
        f1score = f1_score(t_list, label_list, average="samples")

        return acc, precision, recall, f1score

    def Accuracy(self, y_true, y_pred):
        count = 0
        for i in range(y_true.shape[0]):
            p = sum(numpy.logical_and(y_true[i], y_pred[i]))
            q = sum(numpy.logical_or(y_true[i], y_pred[i]))
            count += p / q
        return count / y_true.shape[0]

    def inference(self, logits, y):

        logits = logits.to("cpu")
        y = y.to("cpu")
        zs = logits.data.numpy()
        ts = y.data.numpy()
        labels = list(map(lambda x: (x >= 0.5).astype(int), zs))

        return labels, ts

    def model_test(self, data, model, test_epoch):

        t_properties = data.test_data_label
        z_properties = model(data.test_data,
                             data.test_batch,
                             data.test_data_maccs,
                             data.test_edge_index_list)

        torch.set_printoptions(precision=2)
        p_properties = z_properties

        p_properties = p_properties.data.to('cpu').numpy()
        t_properties = t_properties.data.to('cpu').numpy()

        p_properties[p_properties < 0.5] = 0
        p_properties[p_properties >= 0.5] = 1

        # total_acc = accuracy_score(t_properties_acc, p_properties_acc)
        total_acc = self.Accuracy(t_properties, p_properties)
        total_precision = precision_score(t_properties, p_properties, average="samples")
        total_recall = recall_score(t_properties, p_properties, average="samples")
        total_f1score = f1_score(t_properties, p_properties, average="samples")

        acc_list = []
        pre_list = []
        rec_list = []
        f1_list = []

        for c in range(11):
            y_true = t_properties[:, c]
            y_pred = p_properties[:, c]

            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1score = f1_score(y_true, y_pred)

            print('Class ' + str(c + 1) + ' statistics:')
            print('Accuracy %.4f, Precision %.4f, Recall %.4f, F1-Score %.4f\n' % (acc,
                                                                                   precision,
                                                                                   recall,
                                                                                   f1score))

            acc_list.append(acc)
            pre_list.append(precision)
            rec_list.append(recall)
            f1_list.append(f1score)

        print("%f\t%f\t%f\t%f" %
              (total_acc,
               total_precision,
               total_recall,
               total_f1score))

        return total_acc, \
               total_precision, \
               total_recall, \
               total_f1score, \
               [acc_list, pre_list, rec_list, f1_list]

    def run_model(self,
                  model,
                  optimizer,
                  loss_fn,
                  data,
                  epochs,
                  test_epoch,
                  test_mode=False):

        total_loss = 0

        print('E '
              '\t\t T '
              '\t\t Loss '
              '\t\t Acc_v '
              '\t\t Pre_v '
              '\t\t Rec_v '
              '\t\t F1_v'
              '\t\t Acc_t '
              '\t\t Pre_t '
              '\t\t Rec_t '
              '\t\t F1_t')

        for epoch in range(1, epochs + 1):

            model.train()

            start = time.time()
            for x, train_batch, global_f, y, edge_index in zip(data.train_data_batch_list,
                                                               data.train_batch_list,
                                                               data.train_maccs_batch_list,
                                                               data.train_label_batch_list,
                                                               data.train_edge_index_batch_list):

                logits = model(x, train_batch, global_f, edge_index)
                loss = loss_fn(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()

            auc_val, precision_val, recall_val, f1score_val = self.evaluate(model,
                                                                            data,
                                                                            mode="val_data")

            auc_test, precision_test, recall_test, f1score_test = self.evaluate(model,
                                                                                data,
                                                                                mode="test_data")
            cost_time = time.time() - start

            print('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (
                  epoch,
                  cost_time,
                  loss.item(),
                  auc_val,
                  precision_val,
                  recall_val,
                  f1score_val,
                  auc_test,
                  precision_test,
                  recall_test,
                  f1score_test))

        if test_mode:

            avg_acc, avg_pre, avg_rec, avg_f1, class_information_list = self.model_test(data, model, test_epoch)

            return avg_acc, avg_pre, avg_rec, avg_f1, class_information_list

        return auc_val, precision_val, recall_val, f1score_val

if __name__=="__main__":

   pass