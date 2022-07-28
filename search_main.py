import os
import time
from pathway_utils import PathWay
from automsr.parallel import ParallelConfig
from automsr.search_space.search_space_config import SearchSpace
from automsr.architecture_search.age.search_algorithm import Search
from automsr.architecture_search.age.utils import save_best_gnn_architecture

ParallelConfig(True)

t1 = time.time()
graph = PathWay(0.8, 0.1, 10, 70)

search_parameter = {"child_num": "1",
                    "mutation_strength": "[1]",
                    "population_K": "5",
                    "parent_num": "2",
                    "search_epoch": "2"}

gnn_parameter = {"drop_out": 0.0,
                 "learning_rate": 0.001,
                 "learning_rate_decay": 0.0005,
                 "train_epoch": 1,
                 "model_select": "min_loss",
                 "retrain_epoch": 1,
                 "one_layer_component_num": 5,
                 "early_stop": False,
                 "early_stop_mode": "val_acc",
                 "early_stop_patience ": 10}

search_space = SearchSpace(gnn_layers=2)

graphpas_instance = Search(graph,
                           search_parameter,
                           gnn_parameter,
                           search_space)

gnn_architecture_list = graphpas_instance.search_operator()
print("the optimal gnn architecture:", gnn_architecture_list)
print("test time consuming:", time.time() - t1, " s")

print("best gnn saving...")
path = os.path.split(os.path.realpath(__file__))[0] + "/data_save/"
if not os.path.exists(path):
    os.makedirs(path)
save_best_gnn_architecture(gnn_architecture_list, path, "the_best_gnn.txt")

# the optimal hp searching
print(35*"+", "the optimal hp search start:", 35*"+")
import numpy as np
from pathway_utils import PathWay
from automsr.automsr_manager.pathway_gnn_manager import PathWayGnnManager
from automsr.automsr_manager import utils
from hyperopt import tpe
from automsr.hp_search.search_manager_pathway import HpSearchObj
import os

for gnn in gnn_architecture_list:
    graph = PathWay(0.8, 0.1, 10, 70)

    gnn_parameter = {"drop_out": 0.0,
                     "learning_rate": 0.001,
                     "learning_rate_decay": 0.0005,
                     "train_epoch": 1,
                     "model_select": "min_loss",
                     "retrain_epoch": 1,
                     "early_stop": False,
                     "early_stop_mode": "val_acc",
                     "early_stop_patience": 10}

    gnn_architecture = gnn

    HP_tuining = HpSearchObj(gnn_architecture, graph, gnn_parameter)
    learning_rate, learning_rate_decay, min_batch, embedding_dim = HP_tuining.hp_tuning(search_epoch=2,
                                                                                        search_algorithm=tpe.suggest)
    print("best learning_rate:", learning_rate)
    print("best learning_rate_decay", learning_rate_decay)
    hp_list = [learning_rate, learning_rate_decay, min_batch, embedding_dim]
    path = os.path.split(os.path.realpath(__file__))[0] + "/data_save/hyparameter"
    if not os.path.exists(path):
        os.makedirs(path)
    utils.hyparameter_save(path, str(gnn) + "_.txt", hp_list)

    train_epoch = 1
    model_select = "min_loss"
    retrain_epoch = 1
    one_layer_component_num = 5
    early_stop = False
    early_stop_mode = "val_acc"
    early_stop_patience = 10
    gcn_architecture = gnn

    acc_list = []
    pre_list = []
    rec_list = []
    f1_list = []
    class_list = []
    for i in range(1):
        data = PathWay(0.8, 0.1, min_batch, embedding_dim)
        model = PathWayGnnManager(0.0,
                                  learning_rate,
                                  learning_rate_decay,
                                  train_epoch,
                                  model_select,
                                  one_layer_component_num,
                                  early_stop,
                                  early_stop_mode,
                                  early_stop_patience)

        model.build_model(gcn_architecture, data)
        avg_acc, avg_pre, avg_rec, avg_f1, class_infor_list = model.train(i, test_mode=True)
        acc_list.append(avg_acc)
        pre_list.append(avg_pre)
        rec_list.append(avg_rec)
        f1_list.append(avg_f1)
        class_list.append(class_infor_list)

    path = os.path.split(os.path.realpath(__file__))[0] + "/data_save/pathway_test_record"
    if not os.path.exists(path):
        os.makedirs(path)
    utils.test_data_record(path,
                           str(gnn) + "_pathway.txt",
                           np.mean(acc_list), np.std(acc_list),
                           np.mean(pre_list), np.std(pre_list),
                           np.mean(rec_list), np.std(rec_list),
                           np.mean(f1_list), np.std(f1_list))
    c1_acc = []
    c2_acc = []
    c3_acc = []
    c4_acc = []
    c5_acc = []
    c6_acc = []
    c7_acc = []
    c8_acc = []
    c9_acc = []
    c10_acc = []
    c11_acc = []

    c1_pre = []
    c2_pre = []
    c3_pre = []
    c4_pre = []
    c5_pre = []
    c6_pre = []
    c7_pre = []
    c8_pre = []
    c9_pre = []
    c10_pre = []
    c11_pre = []

    c1_rec = []
    c2_rec = []
    c3_rec = []
    c4_rec = []
    c5_rec = []
    c6_rec = []
    c7_rec = []
    c8_rec = []
    c9_rec = []
    c10_rec = []
    c11_rec = []

    c1_f1 = []
    c2_f1 = []
    c3_f1 = []
    c4_f1 = []
    c5_f1 = []
    c6_f1 = []
    c7_f1 = []
    c8_f1 = []
    c9_f1 = []
    c10_f1 = []
    c11_f1 = []

    for epoch_class_info in class_list:
        c1_acc.append(epoch_class_info[0][0])
        c2_acc.append(epoch_class_info[0][1])
        c3_acc.append(epoch_class_info[0][2])
        c4_acc.append(epoch_class_info[0][3])
        c5_acc.append(epoch_class_info[0][4])
        c6_acc.append(epoch_class_info[0][5])
        c7_acc.append(epoch_class_info[0][6])
        c8_acc.append(epoch_class_info[0][7])
        c9_acc.append(epoch_class_info[0][8])
        c10_acc.append(epoch_class_info[0][9])
        c11_acc.append(epoch_class_info[0][10])

        c1_pre.append(epoch_class_info[1][0])
        c2_pre.append(epoch_class_info[1][1])
        c3_pre.append(epoch_class_info[1][2])
        c4_pre.append(epoch_class_info[1][3])
        c5_pre.append(epoch_class_info[1][4])
        c6_pre.append(epoch_class_info[1][5])
        c7_pre.append(epoch_class_info[1][6])
        c8_pre.append(epoch_class_info[1][7])
        c9_pre.append(epoch_class_info[1][8])
        c10_pre.append(epoch_class_info[1][9])
        c11_pre.append(epoch_class_info[1][10])

        c1_rec.append(epoch_class_info[2][0])
        c2_rec.append(epoch_class_info[2][1])
        c3_rec.append(epoch_class_info[2][2])
        c4_rec.append(epoch_class_info[2][3])
        c5_rec.append(epoch_class_info[2][4])
        c6_rec.append(epoch_class_info[2][5])
        c7_rec.append(epoch_class_info[2][6])
        c8_rec.append(epoch_class_info[2][7])
        c9_rec.append(epoch_class_info[2][8])
        c10_rec.append(epoch_class_info[2][9])
        c11_rec.append(epoch_class_info[2][10])

        c1_f1.append(epoch_class_info[3][0])
        c2_f1.append(epoch_class_info[3][1])
        c3_f1.append(epoch_class_info[3][2])
        c4_f1.append(epoch_class_info[3][3])
        c5_f1.append(epoch_class_info[3][4])
        c6_f1.append(epoch_class_info[3][5])
        c7_f1.append(epoch_class_info[3][6])
        c8_f1.append(epoch_class_info[3][7])
        c9_f1.append(epoch_class_info[3][8])
        c10_f1.append(epoch_class_info[3][9])
        c11_f1.append(epoch_class_info[3][10])

    c1_acc = np.mean(c1_acc)
    c2_acc = np.mean(c2_acc)
    c3_acc = np.mean(c3_acc)
    c4_acc = np.mean(c4_acc)
    c5_acc = np.mean(c5_acc)
    c6_acc = np.mean(c6_acc)
    c7_acc = np.mean(c7_acc)
    c8_acc = np.mean(c8_acc)
    c9_acc = np.mean(c9_acc)
    c10_acc = np.mean(c10_acc)
    c11_acc = np.mean(c11_acc)

    acc = [c1_acc, c2_acc, c3_acc, c4_acc, c5_acc,
           c6_acc, c7_acc, c8_acc, c9_acc, c10_acc,
           c11_acc]

    c1_pre = np.mean(c1_pre)
    c2_pre = np.mean(c2_pre)
    c3_pre = np.mean(c3_pre)
    c4_pre = np.mean(c4_pre)
    c5_pre = np.mean(c5_pre)
    c6_pre = np.mean(c6_pre)
    c7_pre = np.mean(c7_pre)
    c8_pre = np.mean(c8_pre)
    c9_pre = np.mean(c9_pre)
    c10_pre = np.mean(c10_pre)
    c11_pre = np.mean(c11_pre)

    pre = [c1_pre, c2_pre, c3_pre, c4_pre, c5_pre,
           c6_pre, c7_pre, c8_pre, c9_pre, c10_pre, c11_pre]

    c1_rec = np.mean(c1_rec)
    c2_rec = np.mean(c2_rec)
    c3_rec = np.mean(c3_rec)
    c4_rec = np.mean(c4_rec)
    c5_rec = np.mean(c5_rec)
    c6_rec = np.mean(c6_rec)
    c7_rec = np.mean(c7_rec)
    c8_rec = np.mean(c8_rec)
    c9_rec = np.mean(c9_rec)
    c10_rec = np.mean(c10_rec)
    c11_rec = np.mean(c11_rec)

    rec = [c1_rec, c2_rec, c3_rec, c4_rec, c5_rec,
           c6_rec, c7_rec, c8_rec, c9_rec, c10_rec, c11_rec]

    c1_f1 = np.mean(c1_f1)
    c2_f1 = np.mean(c2_f1)
    c3_f1 = np.mean(c3_f1)
    c4_f1 = np.mean(c4_f1)
    c5_f1 = np.mean(c5_f1)
    c6_f1 = np.mean(c6_f1)
    c7_f1 = np.mean(c7_f1)
    c8_f1 = np.mean(c8_f1)
    c9_f1 = np.mean(c9_f1)
    c10_f1 = np.mean(c10_f1)
    c11_f1 = np.mean(c11_f1)

    f1 = [c1_f1, c2_f1, c3_f1, c4_f1, c5_f1,
          c6_f1, c7_f1, c8_f1, c9_f1, c10_f1, c11_f1]

    path = os.path.split(os.path.realpath(__file__))[0] + "/data_save/pathway_test_class_record" + str(gnn)
    if not os.path.exists(path):
        os.makedirs(path)
    utils.class_test_data_record(path, "acc.txt", acc)
    utils.class_test_data_record(path, "pre.txt", pre)
    utils.class_test_data_record(path, "rec.txt", rec)
    utils.class_test_data_record(path, "f1.txt", f1)
