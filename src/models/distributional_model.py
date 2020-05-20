import datetime
import sys
import os
import numpy as np
import pickle
import time
import my_utils

class DistributionalModel:

    #############################################################################################################################
    def __init__(self):
        self.corpus = None
        
        self.num_vocab = None
        self.vocab_list = None
        self.vocab_index_dict = None
        self.onehot_vocab_matrix = None

        self.stop_list_path = None
        self.stop_list = None

        self.vocab_embedding_matrix = None
        self.normalization_method = None
        self.reduction_method = None
        self.reduction_size = None

        self.similarity_metric = None
        self.similarity_matrix = None
    
    #############################################################################################################################
    def create_model_name(self, model_type='dm'):
        self.start_datetime = datetime.datetime.timetuple(datetime.datetime.now())
        self.model_name = "{}_{}_{}_{}_{}_{}_{}".format(model_type, self.start_datetime[1],
                                               self.start_datetime[2],
                                               self.start_datetime[3],
                                               self.start_datetime[4],
                                               self.start_datetime[5],
                                               self.start_datetime[6])
        self.model_path = '../models/' + self.model_name

    #############################################################################################################################
    def create_model_directory(self):
        if not os.path.isdir("../models/"):
            os.mkdir("../models/")

        if os.path.isdir(self.model_path):
            print("Model {} already exists".format(self.model_path))
            sys.exit()
        else:
            os.mkdir(self.model_path)

    #############################################################################################################################
    def create_stop_list(self):
        if self.stop_list_path is not None:
            # here is where you would create a stop list from a file
            self.stop_list = []
        else:
            self.stop_list = []

    #############################################################################################################################
    def create_vocabulary(self, num_vocab):
        self.num_vocab = 0
        self.vocab_list = []
        self.vocab_index_dict = {}

        if num_vocab > self.corpus.num_types:
            print("WARNING: num_vocab > corpus.num_types. Setting num_vocab = corpus.num_types")
            num_vocab = self.corpus.num_types
        
        i = 0
        while self.num_vocab < num_vocab:
            current_corpus_type = self.corpus.type_list[i]

            if current_corpus_type == 'UNKNOWN':
                print("ERROR: Corpus type 'UNKNOWN' conflicts with reserved type")
                sys.exit()
            
            if current_corpus_type not in self.stop_list:
                self.vocab_list.append(current_corpus_type)
                self.vocab_index_dict[current_corpus_type] = self.num_vocab
                self.num_vocab += 1
            i += 1

            if i+1 > len(self.corpus.type_list):
                print("Warning: Failed to find {} types to put in ")
                break
        
        self.vocab_list.append("UNKNOWN")
        self.vocab_index_dict['UNKNOWN'] = self.num_vocab
        self.num_vocab += 1

    #############################################################################################################################s
    def create_model_config_files(self):
        f = open(self.model_path+'/config.txt', 'w')
        f.write("model_name: {}\n".format(self.model_name))
        f.write("corpus_name: {}\n".format(self.corpus.corpus_name))
        f.write("num_vocab: {}\n".format(self.num_vocab))
        f.write("stop_list_path: {}\n".format(self.stop_list_path))
        f.close()

        f = open(self.model_path+'/vocab.txt', 'w')
        for i in range(self.num_vocab):
            f.write("{},{}\n".format(i, self.vocab_list[i]))
        f.close()

    #############################################################################################################################
    def normalize_embedding_matrix(self):
        print("    Normalizing using method:", self.normalization_method)
        if self.normalization_method == None:
            pass
        elif self.normalization_method == 'rowsums':
            self.vocab_embedding_matrix = my_utils.row_sum_normalize(self.vocab_embedding_matrix)
        elif self.normalization_method == 'columnsums':
            self.vocab_embedding_matrix = my_utils.column_sum_normalize(self.vocab_embedding_matrix)
        elif self.normalization_method == 'rowzscore':
            self.vocab_embedding_matrix = my_utils.row_zscore_normalize(self.vocab_embedding_matrix)
        elif self.normalization_method == 'columnzscore':
            self.vocab_embedding_matrix = my_utils.column_zscore_normalize(self.vocab_embedding_matrix)
        elif self.normalization_method == 'rowlogentropy':
            self.vocab_embedding_matrix = my_utils.row_log_entropy_normalize(self.vocab_embedding_matrix)
        elif self.normalization_method == 'ppmi':
            self.vocab_embedding_matrix = my_utils.ppmi_normalize(self.vocab_embedding_matrix)
        else:
            raise AttributeError('Invalid arg {} to normalization_method'.format(self.normalization_method))

    #############################################################################################################################
    def reduce_embedding_matrix(self):
        print("Reducing embedding matrix using method", self.reduction_method)
        if self.reduction_method is None:
            pass
        elif self.reduction_method == 'svd':
            row_dimension_loadings, singular_values, column_dimension_loadings = np.linalg.svd(self.vocab_embedding_matrix)
            self.vocab_embedding_matrix = row_dimension_loadings[:,:self.reduction_size]
            np.savetxt(self.model_path+'/svd_singular_values.csv', singular_values, delimiter=",", fmt="%0.4f")
        else:
            raise AttributeError('Invalid arg {} to reduction_method'.format(self.reduction_method))

    #############################################################################################################################
    def save_model(self):
        pickle_file = open(self.model_path + '/model_object.p', 'wb')
        pickle.dump(self, pickle_file)
        pickle_file.close()

    #############################################################################################################################
    def save_embedding_matrix(self):
        np.savetxt(self.model_path+'/embedding_matrix.csv', self.vocab_embedding_matrix, delimiter=",", fmt="%0.4f")
    
    #############################################################################################################################
    def load_model(self, model_name):
        pickle_file = open('../models/'+model_name+'/model_object.p', 'rb')
        instance_object = pickle.load(pickle_file)
        pickle_file.close()
        self.__dict__ = instance_object.__dict__
    
    #############################################################################################################################
    def compute_full_similarity_matrix(self, similarity_metric):
        sim_file_path = self.model_path + '/similarities_' + similarity_metric + '.csv'
        if os.path.exists(sim_file_path):
            raise AttributeError('{} Similarity file for model {} already exists'.format(similarity_metric, self.model_name))

        print("Computing all similarities")
        self.similarity_metric = similarity_metric
        self.similarity_matrix = np.zeros([self.num_vocab, self.num_vocab])
        for i in range(self.num_vocab):
            a = self.vocab_embedding_matrix[i, :]
            for j in range(self.num_vocab):
                if i == j:
                    self.similarity_matrix[i, j] = 1
                else:
                    if i < j:
                        b = self.vocab_embedding_matrix[j, :]
                        if similarity_metric == 'cosine':
                            self.similarity_matrix[i, j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                            self.similarity_matrix[j, i] = self.similarity_matrix[i, j]
                        elif similarity_metric == 'euclidean':
                                # ((x1 - x2)^m + (y1 - y2)^m )^(1/m)
                            self.similarity_matrix[i, j] = np.linalg.norm(a-b, ord=2)
                            self.similarity_matrix[j, i] = self.similarity_matrix[i, j]
                        elif similarity_metric == 'cityblock':
                                # ((x1 - x2)^m + (y1 - y2)^m )^(1/m)
                            self.similarity_matrix[i, j] = np.linalg.norm(a-b, ord=1)
                            self.similarity_matrix[j, i] = self.similarity_matrix[i, j]
                        elif similarity_metric == 'correlation':
                                # ((x1 - x2)^m + (y1 - y2)^m )^(1/m)
                            self.similarity_matrix[i, j] = np.corrcoef(a,b)[0,1]
                            self.similarity_matrix[j, i] = self.similarity_matrix[i, j]
                        else:
                            raise AttributeError('Invalid arg "similarity_metric".')
            if i % 10 == 0:
                print("    Finished {} rows".format(i))
        np.savetxt(sim_file_path, self.similarity_matrix, delimiter=",", fmt="%0.4f")
    
    #############################################################################################################################
    def get_nearest_neighbors(self, n, sim_file_path=None):
        print("Getting {} nearest neighbors".format(n))

        if self.similarity_matrix is None and sim_file_path is None:
            raise AttributeError('No similarity matrix for this model exists. Load or create one.')
        if self.similarity_matrix is None:
            raise AttributeError('No similarity matrix exists. Load sim matrix functionality not implimented".')
        if self.similarity_metric is None:
            raise AttributeError('Similarity metric cannot be None')
        
        f = open(self.model_path + '/' + self.similarity_metric + '_nearest_neighbors.csv', 'w')
        n += 1 # this is to make sure we exclude the word's most similar word, itself
        if self.similarity_matrix is None:
            raise AttributeError('Must compute similarity matrix before computing nearest neighbors".')
        else:
            for i in range(self.num_vocab):
                sims = self.similarity_matrix[i,:]
                largest_indexes = np.argpartition(sims, -n)[-n:]
                sorted_largest_indexes = largest_indexes[np.argsort(sims[largest_indexes])]
                for j in range(n):
                    neighbor_index = sorted_largest_indexes[-(j+1)]
                    if i != neighbor_index:
                        f.write("{},{},{:0.4f}\n".format(self.vocab_list[i], self.vocab_list[neighbor_index], sims[neighbor_index]))
        f.close()