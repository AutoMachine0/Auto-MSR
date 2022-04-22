import os
import numpy as np
# from autognas.search_space import search_space, stack_gnn_architecture

##### 数据读写 start #####
def hyparameter_save(path, file_name, hp_list):

    with open(path + "/" + file_name, "w") as f:
        for hp in hp_list:
            f.write(str(hp)+"\n")
    print("hyparameter save done !")

def experiment_initial_data_save(path, file_name, gnn_architecture_list, acc_list):
    with open(path + "/" + file_name, "w") as f:
        for gnn_architecture,  val_acc, in zip(gnn_architecture_list, acc_list):
            f.write(str(gnn_architecture)+";"+str(val_acc)+"\n")
    print("data save done !")

def experiment_graphpas_final_data_save(path, file_name, gnn, test_acc):
    with open(path + "/" + file_name, "w") as f:
        f.write(str(gnn)+"\n"+str(test_acc))
    print("test data save done !")

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
    print("search time save done !")

def experiment_time_save_initial(path, file_name, time_cost):
    with open(path + "/" + file_name, "w") as f:
        f.write(str(time_cost)+"\n")
    print("initial time save done !")
##### 数据读写 end #####

##### 路径获取 start #####
def path_get():
    # 当前文件目录
    c_path = os.path.abspath('')
    return c_path
##### 路径获取 end #####

##### 变异选择概率计算 start #####
def mutation_selection_probability(sharing_population, gnn_architecture_flow):

    p_list = []
    for i in range(len(gnn_architecture_flow)):
        p_list.append([])

    while sharing_population:
        gnn = sharing_population.pop()
        for index in range(len(p_list)):
            p_list[index].append(gnn[index])

    gene_information_entropy = []
    for sub_list in p_list:
        gene_information_entropy.append(information_entropy(sub_list))

    exp_x = np.exp(gene_information_entropy)
    probability = exp_x / np.sum(exp_x)

    return probability

def information_entropy(p_list):
    dict = {}
    length = len(p_list)
    for key in p_list:
        dict[key] = dict.get(key, 0) + 1

    p_list = []
    for key in dict:
        p_list.append(dict[key] / length)

    p_array = np.array(p_list)
    log_p = np.log2(p_array)
    information_entropy = -sum(p_array * log_p)

    return information_entropy
##### 变异选择概率计算 end #####

##### 基于fitness的top population选择  start #####
def top_population_select(population, accuracy, top_k):
    population_dict = {}
    for key, value in zip(population, accuracy):
        population_dict[str(key)] = value
    # 按acc排序
    rank_population_dict = sorted(population_dict.items(), key=lambda x: x[1], reverse=True)

    sharing_popuplation = []
    sharing_validation_acc = []

    i = 0
    for key, value in rank_population_dict:

        if i == top_k:
            break
        else:
            sharing_popuplation.append(eval(key))
            sharing_validation_acc.append(value)
            i += 1
    return sharing_popuplation, sharing_validation_acc
##### 基于fitness的top population选择  end #####

# ##### gnn architectrue 编码解码 start #####
# def gnn_architecture_embedding_decoder(gnn_architecture_embedding):
#     gnn_architecture = []
#     for component_embedding, component_name in zip(gnn_architecture_embedding, stack_gnn_architecture):
#         component = search_space[component_name][component_embedding]
#         gnn_architecture.append(component)
#     print('gnn_architecture: ', gnn_architecture)
#     return gnn_architecture
#
# def random_generate_gnn_architecture_embedding():
#     gnn_architecture_embedding = []
#     for component in stack_gnn_architecture:
#         gnn_architecture_embedding.append(np.random.randint(0, len(search_space[component])))
#     print('gnn_architecture_embedding:', gnn_architecture_embedding)
#     return gnn_architecture_embedding
# ##### gnn architectrue 编码解码 end #####

if __name__=="__main__":
    print(path_get())