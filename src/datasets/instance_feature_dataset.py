import numpy as np
import os
import sys
import pickle
import random
import config
import datetime
import my_utils


class InstanceFeatureDataset:
    #############################################################################################################################
    def __init__(self):
        self.dataset_type = 'Instance_Feature'
        self.dataset_name = None
        self.dataset_path = None

        self.num_categories = None
        self.category_size_list = None
        self.category_list = None
        self.category_index_dict = None

        self.num_instances = None
        self.instance_list = None
        self.instance_index_dict = None

        self.num_features = None
        self.feature_list = None
        self.feature_index_dict = None

        self.instance_category_dict = None
        self.instance_category_matrix = None
        self.instance_feature_matrix = None

        self.num_folds = None
        self.training_fold_list = None
        

    #############################################################################################################################
    def __str__(self):
        output_string = "Instance Feature Dataset"
        return output_string

    #############################################################################################################################
    def create_dataset(self):
        
        self.num_categories = 2
        self.category_size_list = [2, 2]
        self.num_instances = 4
        self.num_features = 2
        self.category_list = ['a', 'b']
        self.category_index_dict = {'a': 0, 'b': 1}
        self.instance_list = ['a1', 'a2', 'b1', 'b2']
        self.instance_index_dict = {'a1': 0, 'a2': 1, 'b1': 2, 'b2': 3}
        self.feature_list = ['F1', 'F2']
        self.feature_index_dict = {'F1': 0, 'F2': 1}
        self.instance_category_dict = {'a1': 'a', 'a2': 'a', 'b1': 'b', 'b2': 'b'}

        self.name_dataset()
        self.create_dataset_directory()
        self.generate_instance_feature_data()
        self.save_dataset()

    #############################################################################################################################
    def name_dataset(self, name='ifd'):
        self.start_datetime = datetime.datetime.timetuple(datetime.datetime.now())
        self.dataset_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(name, 
                                               self.num_categories, self.num_instances, self.num_features,
                                               self.start_datetime[1],
                                               self.start_datetime[2],
                                               self.start_datetime[3],
                                               self.start_datetime[4],
                                               self.start_datetime[5],
                                               self.start_datetime[6])
        self.dataset_path = '../datasets/'+self.dataset_name
    
    #############################################################################################################################
    def create_dataset_directory(self):
        if not os.path.isdir("../datasets/"):
            os.mkdir("../datasets/")

        if os.path.isdir(self.dataset_path):
            print("Dataset {} already exists".format(self.dataset_path))
            sys.exit()
        else:
            os.mkdir(self.dataset_path)

    #############################################################################################################################
    def generate_instance_feature_data(self):
        self.instance_feature_matrix = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], float)

    #############################################################################################################################
    def save_dataset(self):
        f = open(self.dataset_path+'/categories.csv', 'w')
        for i in range(self.num_categories):
            current_category = self.category_list[i]
            output_string = "{},{}\n".format(i,current_category)
            f.write(output_string)
        f.close()

        f = open(self.dataset_path+'/instances.csv', 'w')
        for i in range(self.num_instances):
            current_instance = self.instance_list[i]
            category = self.instance_category_dict[current_instance]
            output_string = "{},{},{}\n".format(i,category,current_instance)
            f.write(output_string)
        f.close()

        f = open(self.dataset_path+'/features.csv', 'w')
        for i in range(self.num_features):
            output_string = "{},{}\n".format(i, self.feature_list[i])
            f.write(output_string)
        f.close()

        np.savetxt(self.dataset_path+'/data_matrix.csv', self.instance_feature_matrix, delimiter=",", fmt="%0.4f")
        
        pickle_file = open(self.dataset_path + '/dataset_object.p', 'wb')
        pickle.dump(self, pickle_file)
        pickle_file.close()

    #############################################################################################################################
    def load_dataset(self, dataset_name):
        pickle_file = open('../datasets/'+dataset_name+'/dataset_object.p', 'rb')
        instance_object = pickle.load(pickle_file)
        pickle_file.close()
        self.__dict__ = instance_object.__dict__
        if self.num_categories > 0:
            self.create_instance_category_matrix()

    #############################################################################################################################
    def create_training_folds(self, num_folds=config.InstanceFeatureDataset.num_folds, remove_unknowns=True):
        shuffled_instances = self.instance_list.copy()
        random.shuffle(shuffled_instances)

        self.training_fold_list = []
        self.num_folds = num_folds

        k, m = divmod(self.num_instances, self.num_folds)
        self.training_fold_list = list((shuffled_instances[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(self.num_folds)))
    
    #############################################################################################################################
    def create_instance_category_matrix(self):
        self.instance_category_matrix = np.zeros([self.num_instances, self.num_categories], float)
        
        for i in range(self.num_instances):
            category = self.instance_category_dict[self.instance_list[i]]
            category_index = self.category_index_dict[category]
            self.instance_category_matrix[i, category_index] = 1
    
    #############################################################################################################################
    def normalize_instance_feature_matrix(self, normalization_method):
        if normalization_method == None:
            pass
        elif normalization_method == 'rowsums':
            self.instance_feature_matrix = my_utils.row_sum_normalize(self.instance_feature_matrix)
        elif normalization_method == 'columnsums':
            self.instance_feature_matrix = my_utils.column_sum_normalize(self.instance_feature_matrix)
        elif normalization_method == 'rowzscore':
            self.instance_feature_matrix = my_utils.row_zscore_normalize(self.instance_feature_matrix)
        elif normalization_method == 'columnzscore':
            self.instance_feature_matrix = my_utils.column_zscore_normalize(self.instance_feature_matrix)
        elif normalization_method == 'rowlogentropy':
            self.instance_feature_matrix = my_utils.row_log_entropy_normalize(self.instance_feature_matrix)
        elif normalization_method == 'tfidf':
            self.instance_feature_matrix = my_utils.tfdif_normalize(self.instance_feature_matrix)
        elif normalization_method == 'ppmi':
            self.instance_feature_matrix = my_utils.ppmi_normalize(self.instance_feature_matrix)
        else:
            print("ERROR: Unrecognized normalization_method", self.normalization_method)
            sys.exit()
        
    #############################################################################################################################
    def svd_instance_feature_matrix(self):
        self.row_dimension_loadings, self.singular_values, self.column_dimension_loadings = np.linalg.svd(self.instance_feature_matrix)
        f = open(self.dataset_path+"/singular_values.txt", 'w')
        print(self.singular_values)
        for i in range(len(self.singular_values)):
            f.write(str(self.singular_values[i]) + '\n')
        f.close()

        np.savetxt(self.dataset_path+'/row_dimension_loadings.csv', self.row_dimension_loadings, delimiter=",", fmt="%0.4f")
        np.savetxt(self.dataset_path+'/column_dimension_loadings.csv', self.column_dimension_loadings, delimiter=",", fmt="%0.4f")
