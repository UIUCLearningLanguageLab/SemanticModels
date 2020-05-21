import numpy as np
from models import classifier
import config
import datetime
import os
import sys
import random

class LogisticRegressionClassifier(classifier.Classifier):
    #############################################################################################################################
    def __init__(self, dataset, all_folds=True, verbose=False, 
                 learning_rate=config.LogisticRegressionClassifier.learning_rate,
                 num_epochs=config.LogisticRegressionClassifier.num_epochs,
                 weight_init_stdev=config.LogisticRegressionClassifier.weight_init_stdev,
                 output_freq=config.LogisticRegressionClassifier.output_freq,
                 save_f1_history=config.LogisticRegressionClassifier.save_f1_history,
                 save_ba_history=config.LogisticRegressionClassifier.save_ba_history):

        super().__init__(dataset, all_folds, verbose)

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_init_stdev = weight_init_stdev
        self.output_freq = output_freq
        self.save_f1_history = save_f1_history
        self.save_ba_history = save_ba_history

        self.input_size = self.dataset.num_features
        self.output_size = self.dataset.num_categories
        self.y_bias_list = []
        self.y_x_weight_list = []

        print("\nTraining Logistic Regression Classifier of Size {}-{} on {} for {} epochs".format(self.input_size, self.output_size, 
                                                                                                  self.dataset.dataset_name, 
                                                                                                  self.num_epochs))

        self.create_model_name("lrc")
        self.create_model_directory()
        self.create_model_config_files()
        self.add_model_config_details()

        self.initialize_weights()
        self.train()
        self.create_confusion_matrix()
        self.save_confusion_matrix()

        self.compute_performance_summary()
        self.save_performance_summary()
        self.save_model()
        self.save_full_results()

    #############################################################################################################################
    def add_model_config_details(self):
        f = open(self.model_path + '/config.csv', 'a')
        f.write("learning_rate: {}".format(self.learning_rate))
        f.write("num_epochs: {}".format(self.num_epochs))
        f.write("weight_init_stdev: {}".format(self.weight_init_stdev))
        f.close()

    #############################################################################################################################
    def initialize_weights(self):
        for i in range(self.dataset.num_folds):
            self.y_bias_list.append(np.random.normal(0, self.weight_init_stdev, [self.output_size]))
            self.y_x_weight_list.append(np.random.normal(0, self.weight_init_stdev, [self.output_size, self.input_size]))

    #############################################################################################################################
    def train(self):
        print("    Training for {} Epochs".format(self.num_epochs))
        sse_matrix = np.zeros([self.num_epochs, self.dataset.num_folds])
        if self.save_ba_history or self.save_f1_history:
            self.calculate_and_save_current_epoch(-1)

        for i in range(self.num_epochs):

            for j in range(self.dataset.num_folds):
                current_fold = self.dataset.training_fold_list[j].copy()
                random.shuffle(current_fold)
                y_x_weights = np.copy(self.y_x_weight_list[j])
                y_bias = np.copy(self.y_bias_list[j])

                if i != j:

                    for k in range(len(current_fold)):
                        current_instance = current_fold[k]
                        instance_index = self.dataset.instance_index_dict[current_instance]
                        current_category = self.dataset.instance_category_dict[current_instance]
                        category_index = self.dataset.category_index_dict[current_category]
                        x = self.dataset.instance_feature_matrix[instance_index, :]
                        y = self.dataset.instance_category_matrix[instance_index, :]
                        o = self.forward(x, y_bias, y_x_weights)
                        cost = self.calculate_cost(y, o)
                        sse_matrix[i,j] += (cost**2).sum()
                        y_bias, y_x_weights = self.update_weights(x, o, cost, y_bias, y_x_weights)
                    
                    self.y_x_weight_list[j] = y_x_weights
                    self.y_bias_list[j] = y_bias
                    
            if (i+1) % self.output_freq == 0:
                print("        Finished Epoch", i+1, sse_matrix[i, :])
                if self.save_ba_history or self.save_f1_history:
                    self.calculate_and_save_current_epoch(i)
    
    ############################################################################################################
    def create_confusion_matrix(self):
        print("            Testing Model")
        self.confusion_matrix = np.zeros([self.dataset.num_folds, self.dataset.num_categories, self.dataset.num_categories])
        self.full_result_list = []  # [[fold, item, actual_cat, guess_cat, correct, sims], ] 

        for i in range(self.dataset.num_folds):
            current_fold = self.dataset.training_fold_list[i]
            for j in range(len(current_fold)):
                current_instance = current_fold[j]
                instance_index = self.dataset.instance_index_dict[current_instance]
                correct_category = self.dataset.instance_category_dict[current_instance]
                correct_category_index = self.dataset.category_index_dict[correct_category]
                x = self.dataset.instance_feature_matrix[instance_index, :]
                y = self.dataset.instance_category_matrix[instance_index, :]
                o = self.forward(x, self.y_bias_list[i], self.y_x_weight_list[i])
                guess_category_index = np.argmax(o)
                guess_category = self.dataset.category_list[guess_category_index]
                if guess_category_index == correct_category_index:
                    correct = 1
                else:
                    correct = 1
                self.confusion_matrix[i, correct_category_index, guess_category_index] += 1
                self.full_result_list.append([[i, current_instance, correct_category, guess_category, correct], o])
        
    ############################################################################################################
    def forward(self, x, y_bias, y_x_weights):
        z_o = np.dot(y_x_weights, x) + y_bias
        o = 1/(1+np.exp(-z_o))
        return o
    
    ############################################################################################################
    def calculate_cost(self, y, o):
        return y - o

    ############################################################################################################
    def update_weights(self, x, o, cost, y_bias, y_x_weights):
        y_delta = cost * 1/(1+np.exp(-o)) * (1 - 1/(1+np.exp(-o)))
        y_bias += y_delta * self.learning_rate
        y_x_weights += (np.dot(y_delta.reshape(len(y_delta), 1), x.reshape(1, len(x))) * self.learning_rate)
        return y_bias, y_x_weights
