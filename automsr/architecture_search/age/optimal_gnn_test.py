import numpy as np
from pathway_utils import PathWay
from automsr.automsr_manager.pathway_gnn_manager import PathWayGnnManager
from automsr.automsr_manager import utils
import os

def gnn_test(gnn_architecture,gnn_index):

    print("the optimal gnn architecture:", gnn_architecture)
    drop_out = 0.0
    learning_rate = 0.001
    learning_rate_decay = 0.0005
    train_epoch = 1
    model_select = "min_loss"
    one_layer_component_num = 5
    early_stop = False
    early_stop_mode = "val_acc"
    early_stop_patience = 10

    auc_list = []
    pre_list = []
    rec_list = []
    f1_list = []
    class_list = []

    for i in range(1):
        data = PathWay(0.8, 0.1, 20, 70)
        model = PathWayGnnManager(drop_out,
                                  learning_rate,
                                  learning_rate_decay,
                                  train_epoch,
                                  model_select,
                                  one_layer_component_num,
                                  early_stop,
                                  early_stop_mode,
                                  early_stop_patience)

        model.build_model(gnn_architecture, data)
        avg_auc, avg_pre, avg_rec, avg_f1, class_infor_list = model.train(i, test_mode=True)
        auc_list.append(avg_auc)
        pre_list.append(avg_pre)
        rec_list.append(avg_rec)
        f1_list.append(avg_f1)
        class_list.append(class_infor_list)

    path = os.path.split(os.path.realpath(__file__))[0][:-30] + "/data_save/pathway_test_record"

    if not os.path.exists(path):
        os.makedirs(path)
    utils.test_data_record(path,
                           "pathway.txt",
                           np.mean(auc_list), np.std(auc_list),
                           np.mean(pre_list), np.std(pre_list),
                           np.mean(rec_list), np.std(rec_list),
                           np.mean(f1_list), np.std(f1_list))
    c1_auc = []
    c2_auc = []
    c3_auc = []
    c4_auc = []
    c5_auc = []
    c6_auc = []
    c7_auc = []
    c8_auc = []
    c9_auc = []
    c10_auc = []
    c11_auc = []

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
        c1_auc.append(epoch_class_info[0][0])
        c2_auc.append(epoch_class_info[0][1])
        c3_auc.append(epoch_class_info[0][2])
        c4_auc.append(epoch_class_info[0][3])
        c5_auc.append(epoch_class_info[0][4])
        c6_auc.append(epoch_class_info[0][5])
        c7_auc.append(epoch_class_info[0][6])
        c8_auc.append(epoch_class_info[0][7])
        c9_auc.append(epoch_class_info[0][8])
        c10_auc.append(epoch_class_info[0][9])
        c11_auc.append(epoch_class_info[0][10])

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

    c1_auc = np.mean(c1_auc)
    c2_auc = np.mean(c2_auc)
    c3_auc = np.mean(c3_auc)
    c4_auc = np.mean(c4_auc)
    c5_auc = np.mean(c5_auc)
    c6_auc = np.mean(c6_auc)
    c7_auc = np.mean(c7_auc)
    c8_auc = np.mean(c8_auc)
    c9_auc = np.mean(c9_auc)
    c10_auc = np.mean(c10_auc)
    c11_auc = np.mean(c11_auc)

    auc = [c1_auc, c2_auc, c3_auc, c4_auc, c5_auc,
           c6_auc, c7_auc, c8_auc, c9_auc, c10_auc,
           c11_auc]

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

    path = os.path.split(os.path.realpath(__file__))[0][:-30] + "/data_save/pathway_test_class_record"
    if not os.path.exists(path):
        os.makedirs(path)
    utils.class_test_data_record(path, "acc.txt", auc)
    utils.class_test_data_record(path, "pre.txt", pre)
    utils.class_test_data_record(path, "rec.txt", rec)
    utils.class_test_data_record(path, "f1.txt", f1)

if __name__=="__main__":
    path = os.path.split(os.path.realpath(__file__))[0][:-30] + "/data_save/the_optimal_gnn"
    print(path)