import numpy as np
import math
import config
import sys
import os
from models import classifier
import my_utils


class PrototypeClassifier(classifier.Classifier):
    #############################################################################################################################
    def __init__(self, dataset, all_folds=True, similarity_metric=config.PrototypeClassifier.similarity_metric, verbose=False):
        super().__init__(dataset, all_folds, verbose)
        print("\nCreating Prototype Classifier")
        self.similarity_metric = similarity_metric
        self.prototype_lists = None
        self.full_result_list = None
        self.check_config_errors()
        self.create_model_name("pc")
        self.create_model_directory()
        self.create_model_config_files()
        self.add_model_config_details()
        self.create_prototypes()
        self.save_prototypes()
        self.create_confusion_matrix()
        self.save_confusion_matrix()
        self.compute_performance_summary()
        self.save_performance_summary()
        self.save_model()
        self.save_full_results()

    #############################################################################################################################
    def check_config_errors(self):
        if self.similarity_metric not in ['cosine', 'cityblock', 'euclidean', 'correlation']:
            print("ERROR: Unrecognized similarity metric", self.similarity_metric)
            sys.exit()
        
        if self.dataset.num_folds < 2:
            print("ERROR: Dataset must have at least 2 training folds")
            sys.exit()

    #############################################################################################################################
    def add_model_config_details(self):
        f = open(self.model_path + '/config.csv', 'a')
        f.write("similarity_metric: {}".format(self.similarity_metric))
        f.close()

    #############################################################################################################################
    def create_prototypes(self):
        print("\n    Creating Prototypes for {} training folds".format(self.dataset.num_folds))

        prototype_sums = np.zeros([self.dataset.num_folds, self.dataset.num_categories, self.dataset.num_features])
        prototype_counts = np.zeros([self.dataset.num_folds, self.dataset.num_categories, self.dataset.num_features])

        for i in range(self.dataset.num_folds):
            for j in range(self.dataset.num_folds):
                if i != j:
                    current_fold = self.dataset.training_fold_list[j]
                    for k in range(len(current_fold)):
                        current_instance = current_fold[k]
                        instance_index = self.dataset.instance_index_dict[current_instance]
                        category = self.dataset.instance_category_dict[current_instance]
                        category_index = self.dataset.category_index_dict[category]
                        feature_vector = self.dataset.instance_feature_matrix[instance_index, :]
                        count_vector = np.ones([self.dataset.num_features])
                        prototype_sums[i, category_index, :] += feature_vector
                        prototype_counts[i, category_index, :] += count_vector
        
        self.prototype_matrix = np.zeros([self.dataset.num_folds, self.dataset.num_categories, self.dataset.num_features])

        for i in range(self.dataset.num_folds):
            for j in range(self.dataset.num_categories):
                for k in range(self.dataset.num_features):
                    if prototype_counts[i,j,k] > 0:
                        self.prototype_matrix[i,j,k] = prototype_sums[i,j,k] / prototype_counts[i,j,k]
                    else:
                        self.prototype_matrix[i,j,k] = np.random.randint(1,100)/1000000
    
    #############################################################################################################################
    def save_prototypes(self):
        prototype_averages = self.prototype_matrix.mean(0)
        np.savetxt(self.model_path+'/prototypes.csv', prototype_averages, delimiter=",", fmt="%0.3f")

    #############################################################################################################################
    def create_confusion_matrix(self):
        self.confusion_matrix = np.zeros([self.dataset.num_folds, self.dataset.num_categories, self.dataset.num_categories])
        self.full_result_list = []  # [[fold, item, actual_cat, guess_cat, correct, sims], ] 

        for i in range(self.dataset.num_folds):
            print("    Testing Fold", i)
            current_test_fold = self.dataset.training_fold_list[i]
            current_prototypes = self.prototype_matrix[i, :, :]

            correct_n_array = np.zeros([self.dataset.num_categories])
            actual_n_array = np.zeros([self.dataset.num_categories])
            guess_n_array = np.zeros([self.dataset.num_categories])

            for j in range(len(current_test_fold)):
                
                current_instance = current_test_fold[j]
                instance_index = self.dataset.instance_index_dict[current_instance]
                feature_vector = self.dataset.instance_feature_matrix[instance_index, :]
                correct_category = self.dataset.instance_category_dict[current_instance]
                correct_category_index = self.dataset.category_index_dict[correct_category]
                
                sim_list = []
                for k in range(self.dataset.num_categories):
                    current_category = self.dataset.category_list[k]
                    category_prototype = current_prototypes[k, :]

                    if self.similarity_metric == 'correlation':
                        current_similarity = my_utils.calculate_cosine(feature_vector, category_prototype)
                    elif self.similarity_metric == 'cityblock':
                        current_similarity = 1 / my_utils.calculate_distance(feature_vector, category_prototype, 1)
                    elif self.similarity_metric == 'euclidean':
                        current_similarity = 1 / my_utils.calculate_distance(feature_vector, category_prototype, 2)                   
                    elif self.similarity_metric == 'cosine':
                        current_similarity = my_utils.calculate_correlation(feature_vector, category_prototype)
                    sim_list.append(current_similarity)
                
                    if k == 0:
                        max_similarity = current_similarity
                        guess_category_index = k
                    else:
                        if current_similarity > max_similarity:
                            max_similarity = current_similarity
                            guess_category_index = k
                guess_category = self.dataset.category_list[guess_category_index]
                if correct_category == guess_category:
                    correct = 1
                else:
                    correct = 0
                
                self.full_result_list.append([[i, current_instance, correct_category, guess_category, correct], sim_list])
                self.confusion_matrix[i, correct_category_index, guess_category_index] += 1
