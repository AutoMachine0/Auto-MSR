import os
import time
import copy
import numpy as np
from automsr.architecture_search.age import utils
from automsr.parallel import ParallelOperater

class PerformanceCheck(object):

    def __init__(self, search_space_dict, stack_gcn_architecture, data, gnn_parameter):
        self.data = data
        self.gnn_parameter = gnn_parameter
        self.search_space_dict = search_space_dict
        self.stack_gcn_architecture = stack_gcn_architecture

    def check(self, population, performance):
        print(35 * "=", "zero performance check start", 35 * "=")
        temp_population = []
        temp_performance = []
        for index in range(len(performance)):
            if performance[index] == 0.0:
                continue
            else:
                temp_population.append(population[index])
                temp_performance.append(performance[index])

        print(35 * "=", "delete zero performance gnns: ", len(population)-len(temp_population), 35 * "=")

        if temp_performance == []:
            Sample = ReSample(self.search_space_dict, self.stack_gcn_architecture, self.data, self.gnn_parameter)
            temp_population, temp_performance = Sample.re_sample_gnn()

        print(35 * "=", "zero performance check end", 35 * "=")

        return temp_population, temp_performance

class ReSample(object):

    def __init__(self, search_space_dict, stack_gcn_architecture, data, gnn_parameter):
        self.search_space = search_space_dict
        self.stack_gcn_architecture = stack_gcn_architecture

        self.data = data
        self.gnn_parameter = gnn_parameter
        self.parallel_estimation = ParallelOperater(self.data, self.gnn_parameter)

    def re_sample_gnn(self):
        print(35 * "=", "supplement random search start", 35 * "=")
        while 1:
            gnn_architecture_embedding = utils.random_generate_gnn_architecture_embedding(self.search_space,
                                                                                          self.stack_gcn_architecture)
            gnn_architecture = utils.gnn_architecture_embedding_decoder(gnn_architecture_embedding,
                                                                        self.search_space,
                                                                        self.stack_gcn_architecture)
            gnn_architecture_list = [gnn_architecture]
            gnn_architecture_embedding_list = [gnn_architecture_embedding]
            result = self.parallel_estimation.estimation(gnn_architecture_list)

            pop_list = gnn_architecture_embedding_list
            performance_list = result

            if result[0] != 0.:
                print(35 * "=", "supplement random search end", 35 * "=")
                return pop_list, performance_list

class PopulationInitialization(object):

    def __init__(self,
                 initial_num,
                 search_space_dict,
                 stack_gcn_architecture):

        self.initial_num = initial_num
        self.initial_gnn_architecture_embedding_list = []
        self.initial_gnn_architecture_list = []
        self.search_space = search_space_dict
        self.stack_gcn_architecture = stack_gcn_architecture

    def initialize_random(self):

        print(35*"=", "population initializing based on random strategy", 35*"=")

        while len(self.initial_gnn_architecture_embedding_list) < self.initial_num:

            gnn_architecture_embedding = utils.random_generate_gnn_architecture_embedding(self.search_space,
                                                                                          self.stack_gcn_architecture)
            gnn_architecture = utils.gnn_architecture_embedding_decoder(gnn_architecture_embedding,
                                                                        self.search_space,
                                                                        self.stack_gcn_architecture)
            # gnn architecture genetic embedding based on number
            self.initial_gnn_architecture_embedding_list.append(gnn_architecture_embedding)
            self.initial_gnn_architecture_list.append(gnn_architecture)

        return self.initial_gnn_architecture_embedding_list, self.initial_gnn_architecture_list

class AgeEvolutionSearcn(object):

    def __init__(self,
                 child_num,
                 mutation_strength,
                 search_space_dict,
                 stack_gcn_architecture):

        if len(mutation_strength) != int(child_num):
            raise Exception("child num and mutation strength num did not match", "the child num is:",
                            int(child_num), " and the mutation strength num is:", len(mutation_strength))

        self.child_num = child_num
        self.mutation_strength = mutation_strength

        self.search_space = search_space_dict
        self.stack_gcn_architecture = stack_gcn_architecture

    def search(self, parent):

        if len(parent) > self.child_num:
            child = parent[:self.child_num]
        else:
            child = parent

        child_embedding_len = len(child[0])

        for index in range(len(child)):

            # confirming mutation point in the gnn architecture genetic list based on random strategy.
            position_to_mutate_list = np.random.choice([gene for gene in range(child_embedding_len)],
                                                       self.mutation_strength[index],
                                                       replace=False)

            for mutation_index in position_to_mutate_list:
                mutation_space = self.search_space[self.stack_gcn_architecture[mutation_index]]
                child[index][mutation_index] = np.random.randint(0, len(mutation_space))

        return child

class Search(object):

    def __init__(self,
                 data,
                 search_parameter,
                 gnn_parameter,
                 search_space):

        self.data = data
        self.search_parameter = search_parameter

        # parallel estimation operator initialize
        self.parallel_estimation = ParallelOperater(data,
                                                    gnn_parameter)

        self.gnn_parameter = gnn_parameter

        self.search_space_dict = search_space.space_getter()
        self.stack_gcn_architecture = search_space.stack_gcn_architecture

        self.Checker = PerformanceCheck(self.search_space_dict,
                                        self.stack_gcn_architecture,
                                        self.data,
                                        self.gnn_parameter)

    def search_operator(self):

        print(35 * "=", "age evolution search start", 35 * "=")

        # searcher initializing
        searcher = AgeEvolutionSearcn(int(self.search_parameter["child_num"]),
                                      eval(self.search_parameter["mutation_strength"]),
                                      self.search_space_dict,
                                      self.stack_gcn_architecture)

        # population initializing and fitness calculating
        time_initial = time.time()
        population_initialization = PopulationInitialization(int(self.search_parameter["population_K"]),
                                                             self.search_space_dict,
                                                             self.stack_gcn_architecture)

        initial_gnn_architecture_embedding_list, initial_gnn_architecture_list = population_initialization.initialize_random()

        result = self.parallel_estimation.estimation(initial_gnn_architecture_list)

        pop_K = initial_gnn_architecture_embedding_list
        pop_K_performance = result

        history_model = copy.deepcopy(initial_gnn_architecture_list)
        history_model_performance = copy.deepcopy(pop_K_performance)

        ########
        pop_K, pop_K_performance = self.Checker.check(pop_K, pop_K_performance)
        ########

        time_initial = time.time() - time_initial

        # initial gnn architecture time cost record
        path = os.path.split(os.path.realpath(__file__))[0][:-31] + "logger/age_logger/"

        if not os.path.exists(path):
            os.makedirs(path)
        utils.experiment_time_save_initial(path, self.data.data_name + "_initial_time.txt", time_initial)

        # parent select based on random
        print(35 * "=", "parent population random select", 35 * "=")

        if len(pop_K) > int(self.search_parameter["parent_num"]):

            parent = utils.parent_random_sample(pop_K,
                                                pop_K_performance,
                                                int(self.search_parameter["parent_num"]))
        else:

            parent = utils.parent_random_sample(pop_K,
                                                pop_K_performance,
                                                len(pop_K))
        time_search_list = []
        epoch = []

        for i in range(int(self.search_parameter["search_epoch"])):
            
            time_search = time.time()

            children_embedding = searcher.search(parent)

            # children decoding
            children_architecture = []
            children_val_performance_list = []

            for gnn_architecture_embedding in children_embedding:
                
                gnn_architecture = utils.gnn_architecture_embedding_decoder(gnn_architecture_embedding,
                                                                            self.search_space_dict,
                                                                            self.stack_gcn_architecture)
                children_architecture.append(gnn_architecture)
                history_model.append(gnn_architecture)

            # children parallel estimation
            result = self.parallel_estimation.estimation(children_architecture)

            for performance in result:
                
                children_val_performance_list += [performance]
                history_model_performance += [performance]

            children_embedding, children_val_performance_list = self.Checker.check(children_embedding,
                                                                                      children_val_performance_list)

            children_architecture_len = len(children_embedding)

            # update population
            if len(pop_K) > children_architecture_len:
                del pop_K[-children_architecture_len:]
                del pop_K_performance[-children_architecture_len:]

                pop_K = children_embedding + pop_K
                pop_K_performance = children_val_performance_list + pop_K_performance

            else:
                pop_K = children_embedding
                pop_K_performance = children_val_performance_list

            # parent select based on random
            print(35 * "=", "parent population random select", 35 * "=")

            if len(pop_K) > int(self.search_parameter["parent_num"]):

                parent = utils.parent_random_sample(pop_K,
                                                    pop_K_performance,
                                                    int(self.search_parameter["parent_num"]))
            else:

                parent = utils.parent_random_sample(pop_K,
                                                    pop_K_performance,
                                                    len(pop_K))

            time_search_list.append(time.time()-time_search)
            epoch.append(i+1)

            # population model architecture and val performance record
            path = os.path.split(os.path.realpath(__file__))[0][:-31] + "logger/age_logger/"
            if not os.path.exists(path):
                os.makedirs(path)
            utils.experiment_graphpas_data_save(path,
                                                self.data.data_name + "_search_epoch_" + str(i+1) + ".txt",
                                                pop_K,
                                                pop_K_performance,
                                                self.search_space_dict,
                                                self.stack_gcn_architecture)

        index = history_model_performance.index(max(history_model_performance))

        best_architecture = history_model[index]
        best_performance = max(history_model_performance)
        print("Best GNN Architecture:\n", best_architecture)
        print("Best VAL Performance:\n", best_performance)

        if len(history_model) > 1:
            gnn_list, _ = utils.top_population_select(history_model, history_model_performance, 1)
        else:
            gnn_list, _ = utils.top_population_select(history_model, history_model_performance, len(history_model))

        path = os.path.split(os.path.realpath(__file__))[0][:-31] + "logger/age_logger/"
        if not os.path.exists(path):
            os.makedirs(path)
        utils.experiment_time_save(path,
                                   self.data.data_name + "_search_time.txt",
                                   epoch,
                                   time_search_list)
        return gnn_list

if __name__=="__main__":
    pass