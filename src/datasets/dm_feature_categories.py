from datasets import instance_feature_dataset
import config
import sys
import numpy as np
import os
import pickle


class DMFeatureCategories(instance_feature_dataset.InstanceFeatureDataset):
    #############################################################################################################################
    def __init__(self):
        super().__init__()

        self.distributional_model = None

    #############################################################################################################################
    def create_dataset(self, model_name, category_file_path):
        print("Creating Distributional Model Feature Category Dataset")
        self.dataset_name = "dmfc"
        self.dataset_path = "../datasets/" + self.dataset_name
    
        self.create_dataset_directory() 
        self.load_instance_features(model_name)
        self.load_categories_and_instances(category_file_path)
        self.create_instance_category_matrix()
        self.save_dataset()                 # defined in the base class

    #############################################################################################################################
    def load_instance_features(self, model_name):
        pickle_file = open('../models/'+model_name+'/model_object.p', 'rb')
        self.distributional_model = pickle.load(pickle_file)
        pickle_file.close()

        self.num_features = 0
        self.feature_list = []
        self.feature_index_dict = {}

        self.instance_feature_matrix = self.distributional_model.vocab_embedding_matrix
        self.num_features =  len(self.instance_feature_matrix[0,:])
        for i in range(self.num_features):
            feature_name = 'F' + str(i+1)
            self.feature_list.append(feature_name)
            self.feature_index_dict[feature_name] = i

    #############################################################################################################################
    def load_categories_and_instances(self, category_file_path):

        category_size_dict = {}
        self.num_categories = 0
        self.category_size_list = []
        self.category_list = []
        self.category_index_dict = {}

        self.num_instances = 0
        self.instance_list = []
        self.instance_index_dict = {}
        self.instance_category_dict = {}

        f = open(category_file_path)
        for line in f:
            data = (line.strip().strip('\n').strip()).split()
            instance = data[0]
            category = data[1]
            if category not in self.category_index_dict:
                self.category_index_dict[category] = self.num_categories
                self.category_list.append(category)
                self.num_categories += 1
                category_size_dict[category] = 0
            
            if instance not in self.instance_index_dict:
                self.instance_index_dict[instance] = self.num_instances
                self.instance_list.append(instance)
                self.num_instances += 1
            
            self.instance_category_dict[instance] = category
            category_size_dict[category] += 1
        f.close()

        self.category_size_list = []
        for category in self.category_list:
            self.category_size_list.append(category_size_dict[category])

    #############################################################################################################################
    def __str__(self):
        output_string = "McRae Feature Norm Dataset [{} {} {}]\n".format(self.num_categories,
                                                                              self.num_instances,
                                                                              self.num_features)
        return output_string
