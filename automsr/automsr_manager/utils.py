import os

def hyparameter_save(path, file_name, hp_list):

    with open(path + "/" + file_name, "w") as f:
        for hp in hp_list:
            f.write(str(hp)+"\n")
    print("hyparameter save done !")

def experiment_data_record(path,
                           file_name,
                           auc_list,
                           pre_list,
                           rec_list,
                           f1_list):
    with open(path + "/" + file_name, "w") as f:
        f.write("auc_list:"+str(auc_list) +
                "\n"+"pre_list:"+str(pre_list) +
                "\n"+"rec_list:"+str(rec_list) +
                "\n"+"f1_list:"+str(f1_list))
    print("data record done !")

def gnn_architecture_save(path,
                          file_name,
                          gnn_list):

    with open(path + "/" + file_name, "w") as f:
        for gnn in gnn_list:
            f.write(str(gnn)+"\n")
    print("the optimal gnn architectures save done !")

def test_data_record(path,
                     file_name,
                     avg_auc, std_auc,
                     avg_pre, std_pre,
                     avg_rec, std_rec,
                     avg_f1,  std_f1):

    with open(path + "/" + file_name, "w") as f:
        f.write("avg_auc:"+str(avg_auc) + "/ std_auc:" + str(std_auc) + "\n" +
                "avg_pre:"+str(avg_pre) + "/ std_pre:" + str(std_pre) + "\n" +
                "avg_rec:"+str(avg_rec) + "/ std_rec:" + str(std_rec) + "\n" +
                "avg_f1:"+str(avg_f1) + "/std_f1:" + str(std_f1))

    print("data record done !")

def class_test_data_record(path,
                           file_name,
                           data):

    with open(path + "/" + file_name, "w") as f:
        f.write("result:"+str(data))
    print("data record done !")

def experiment_graphpas_data_save(path,
                                  file_name,
                                  gnn_architecture_list,
                                  acc_list):

    with open(path + "/" + file_name, "w") as f:
        gnn_architecture_list_temp = []
        for gnn_architecture_embedding in gnn_architecture_list:
            gnn_architecture_list_temp.append(gnn_architecture_embedding_decoder(gnn_architecture_embedding))
        for gnn_architecture,  val_acc, in zip(gnn_architecture_list_temp, acc_list):
            f.write(str(gnn_architecture)+";"+str(val_acc)+"\n")
    print("data save done !")

def experiment_graphpas_data_load(path):

    with open(path, "r") as f:
        gnn_architecture = []
        gnn_acc = []
        for line in f.readlines():
            gnn, acc = line.split(";")
            gnn_architecture.append(gnn)
            gnn_acc.append(acc.replace("\n", ""))
    print("data load done !")
    return gnn_architecture, gnn_acc

def experiment_time_save(path, file_name, epoch, time_cost):

    with open(path + "/" + file_name, "w") as f:
        for epoch_, timestamp, in zip(epoch, time_cost):
            f.write(str(epoch_)+";"+str(timestamp)+"\n")
    print("search time record done !")

def experiment_time_save_initial(path, file_name, time_cost):

    with open(path + "/" + file_name, "w") as f:
        f.write(str(time_cost)+"\n")
    print("initial time save done !")

def path_get():
    # 当前文件目录
    c_path = os.path.abspath('')
    return c_path

# select top population based on fitness
def top_population_select(population, accuracy, top_k):
    population_dict = {}
    for key, value in zip(population, accuracy):
        population_dict[str(key)] = value

    # rank based on fitness
    rank_population_dict = sorted(population_dict.items(), key=lambda x: x[1], reverse=True)

    top_popuplation = []
    top_fitness = []
    i = 0
    for key, value in rank_population_dict:

        if i == top_k:
            break
        else:
            top_popuplation.append(eval(key))
            top_fitness.append(value)
            i += 1
    return top_popuplation, top_fitness

if __name__=="__main__":
    pass