from datasets import instance_feature_dataset
import config
import sys
import numpy as np
import os


class McRaeFeatureNorms(instance_feature_dataset.InstanceFeatureDataset):
    #############################################################################################################################
    def __init__(self):
        super().__init__()

    #############################################################################################################################
    def create_dataset(self, path, categories=False):
        print("Creating McRae Feature Norms Dataset in", path)
        self.import_path = "../external_datasets/mcrae_feature_norms"
        self.dataset_name = "mcrae_feature_norms"
        self.dataset_path = "../datasets/" + self.dataset_name
    
        self.create_dataset_directory()
        self.load_instance_features()
        if categories:
            self.load_instance_categories()
        else:
            self.create_unknown_categories()
        self.save_dataset()
    
    #############################################################################################################################
    def create_dataset_directory(self):
        if os.path.isdir(self.dataset_path):
            print("Dataset {} already exists".format(self.dataset_path))
            sys.exit()
        else:
            os.mkdir(self.dataset_path)

    #############################################################################################################################
    def load_instance_features(self):
        self.feature_list = []
        self.feature_index_dict = {}
        self.num_features = 0

        self.instance_list = []
        self.instance_index_dict = {}
        self.num_instances = 0

        f = open(self.import_path + '/instance_features.csv')
        instance_feature_dict = {}
        for line in f:
            data = (line.strip().strip().strip()).split(',')
            current_instance = data[0]
            feature = data[1]
            count = int(data[2])
            if current_instance not in self.instance_index_dict:
                self.instance_index_dict[current_instance] = self.num_instances
                self.instance_list.append(current_instance)
                self.num_instances += 1
            if feature not in self.feature_index_dict:
                self.feature_index_dict[feature] = self.num_features
                self.feature_list.append(feature)
                self.num_features += 1
            instance_feature_dict[(current_instance, feature)] = count
        f.close()

        self.instance_feature_matrix = np.zeros([self.num_instances, self.num_features], float)
        for item in instance_feature_dict:
            instance_index = self.instance_index_dict[item[0]]
            feature_index = self.feature_index_dict[item[1]]
            count = instance_feature_dict[item]
            self.instance_feature_matrix[instance_index, feature_index] = count

    #############################################################################################################################
    def load_instance_categories(self):
        self.num_categories = 0
        self.category_list = []
        self.category_index_dict = {}
        self.instance_category_dict = {}
        self.category_size_list = []

        try:
            f = open(self.import_path + '/instance_categories.csv')
        except:
            print("ERROR: Cannot open file named 'instance_features.csv' in directory", self.dataset_path)
            sys.exit()
        
        category_size_dict = {}
        for line in f:
            data = (line.strip().strip().strip()).split(',')
            category = data[0]
            current_instance = data[1]

            if category not in self.category_index_dict:
                self.category_index_dict[category] = self.num_categories
                self.category_list.append(category)
                category_size_dict[category] = 1
                self.num_categories +=1
            else:
                category_size_dict[category] += 1
            
            self.instance_category_dict[current_instance] = category
        f.close()

        for category in self.category_list:
            self.category_size_list.append(category_size_dict[category])

    #############################################################################################################################
    def create_unknown_categories(self):
        self.num_categories = 1
        self.category_list = ['UNKNOWN']
        self.category_index_dict = {'UNKNOWN': 0}
        self.category_size_list = [self.num_instances]
        self.instance_category_dict = {}
        for current_instance in self.instance_list:
            self.instance_category_dict[current_instance] = 'UNKNOWN'

    #############################################################################################################################
    def __str__(self):
        output_string = "McRae Feature Norm Dataset [{} {} {}]\n".format(self.num_categories,
                                                                              self.num_instances,
                                                                              self.num_features)
        return output_string
