from datasets import instance_feature_dataset
import config
import string
import numpy as np
import random
import os
import sys
import datetime

class SimpleFeatureCategories(instance_feature_dataset.InstanceFeatureDataset):
    #############################################################################################################################
    def __init__(self):
        super().__init__()
        self.global_mean = None
        self.global_stdev = None
        self.within_stdev = None


    ############################################################################################################################
    def create_dataset(self, num_categories=config.SimpleFeatureCategories.num_categories,
                             num_features=config.SimpleFeatureCategories.num_features,
                             global_mean=config.SimpleFeatureCategories.global_mean,
                             global_stdev=config.SimpleFeatureCategories.global_stdev,
                             within_stdev=config.SimpleFeatureCategories.within_stdev,
                             category_size_list=config.SimpleFeatureCategories.category_size_list):
        
        # init the dataset parameters from the config file
        self.num_categories = num_categories
        self.num_features = num_features
        self.global_mean = global_mean
        self.global_stdev = global_stdev
        self.within_stdev = within_stdev
        self.category_size_list = category_size_list

        self.check_category_sizes()  # check to make sure the category sizes are appropriate
        self.num_instances = sum(self.category_size_list)
        self.create_feature_list()
        self.name_dataset()
        self.create_dataset_directory()
        self.create_instance_category_labels() # create the list of instance labels and the instance-category dictionary
        self.generate_instance_feature_data()
        self.save_dataset()

        self.create_instance_category_matrix()

    ############################################################################################################################
    def check_category_sizes(self):
        # set to 26 if more than 26 categories
        if self.num_categories > 26:
            self.num_categories = 26
            print("Setting num_categories to max=26")

        # create the category size list
        category_sizes = config.SimpleFeatureCategories.category_size_list

        if isinstance(category_sizes, list):
            if len(category_sizes) != self.num_categories:
                print("ERROR: Category Size list is != Num Categories ({},{}".format(len(self.category_size_list), self.num_categories))
            else:
                self.category_size_list = category_sizes
        elif isinstance(category_sizes, int):
            if category_sizes > 0:

                self.category_size_list = []
                for i in range(self.num_categories):
                    self.category_size_list.append(category_sizes)
            else:
                print("ERROR: Category size list must be either a positive integer or a list of positive integers")
        else:
            print("ERROR: Category Size list must be either a positive integer or a list of positive integers")
    
    ############################################################################################################################
    def create_feature_list(self):
        self.feature_list = []
        for i in range(self.num_features):
            self.feature_list.append("F" + str(i+1))

    ############################################################################################################################
    def name_dataset(self):
        # name the dataset based on its parameters and the system time
        self.start_datetime = datetime.datetime.timetuple(datetime.datetime.now())
        self.dataset_name = "sfc_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.num_categories, self.num_instances, self.num_features,
                                               self.start_datetime[1],
                                               self.start_datetime[2],
                                               self.start_datetime[3],
                                               self.start_datetime[4],
                                               self.start_datetime[5],
                                               self.start_datetime[6])
        self.dataset_path = '../datasets/' + self.dataset_name

    ############################################################################################################################
    def create_dataset_directory(self):
        # create the directory if it doesnt already exist
        if os.path.isdir(self.dataset_path):
            print("Dataset {} already exists".format(self.dataset_path))
            sys.exit()
        else:
            os.mkdir(self.dataset_path)

    ############################################################################################################################
    def create_instance_category_labels(self):
        letters = string.ascii_lowercase
        self.category_list = []
        self.category_index_dict = {}
        self.instance_list = []
        self.instance_index_dict = {}
        self.instance_category_dict = {}
        instance_counter = 0
        for i in range(self.num_categories):
            category = letters[i]
            category_size = self.category_size_list[i]
            self.category_list.append(category)
            self.category_index_dict[category] = i
            for j in range(category_size):
                label = category + str(j+1)
                self.instance_list.append(label)
                self.instance_index_dict[label] = instance_counter
                self.instance_category_dict[label] = letters[i]
                instance_counter += 1

    #############################################################################################################################
    def generate_instance_feature_data(self):
        self.instance_feature_matrix = np.zeros([self.num_instances, self.num_features], float)
        instance_counter = 0
        for i in range(self.num_categories):
            category_size = self.category_size_list[i]
            category_feature_means = np.random.normal(self.global_mean, self.global_stdev, self.num_features)
            for j in range(category_size):
                for k in range(self.num_features):
                    self.instance_feature_matrix[instance_counter, k] = np.random.normal(category_feature_means[k], self.within_stdev)
                instance_counter += 1

    #############################################################################################################################
    def __str__(self):
        output_string = "Simple Features Categories Dataset [{} {} {} {} {} {}]\n".format(self.num_categories,
                                                                              self.num_instances,
                                                                              self.num_features,
                                                                              self.global_mean,
                                                                              self.global_stdev,
                                                                              self.within_stdev)
        return output_string