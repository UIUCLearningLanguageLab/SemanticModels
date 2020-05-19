import config
import os
import sys
import numpy as np
import operator
import pickle

class Corpus:

    def __init__(self):
        self.corpus_name = None
        self.import_path = None
        self.corpus_path = None

        self.num_documents = None
        self.document_name_list = None
        self.document_index_dict = None
        self.document_token_list = None
        self.document_size_array = None

        self.num_types = None
        self.type_list = None
        self.type_index_dict = None
        self.type_freq_dict = None
        self.sort_types_method = None

        self.num_tokens = None

    def create_corpus_object(self, corpus_name, import_path, document_names_file=None, sort_types_method=config.Corpus.sort_types_method):
        self.corpus_name = corpus_name
        self.import_path = import_path
        self.document_names_file = document_names_file
        self.sort_types_method = sort_types_method
        self.corpus_path = "../corpora/" + self.corpus_name

        self.create_corpus_directory()
        self.get_document_name_list()
        self.import_corpus()
        self.sort_types()
        self.save_corpus()

    def create_corpus_directory(self):
        if os.path.isdir(self.corpus_path):
            print("Dataset {} already exists".format(self.corpus_path))
            sys.exit()
        else:
            os.mkdir(self.corpus_path)

    def get_document_name_list(self):
        # TODO this will cause problems if there are blank lines in the corpus, because num_documents will be different
        # than the real num documents

        self.document_name_list = []
        f = open(self.import_path+'/corpus.txt')
        self.num_documents = 0
        self.document_index_dict = {}

        if self.document_names_file is None:
            for line in f:
                name = "doc" + str(self.num_documents)
                self.document_name_list.append(name)
                self.document_index_dict[name] = self.num_documents
                self.num_documents += 1

    def import_corpus(self):
        print("Importing corpus")
        
        self.document_token_list = []
        self.document_size_array = np.zeros([self.num_documents])

        self.num_types = 0
        self.type_list = []
        self.type_index_dict = {}
        self.type_freq_dict = {}

        self.num_tokens = 0

        f = open(self.import_path+'/corpus.txt')
        f_out = open(self.corpus_path+'/corpus.txt', 'w')
        i = 0
        for line in f:
            token_list = self.get_cleaned_token_list(line)
            if len(token_list) > 0:
                f_out.write(' '.join(token_list)+'\n')
                self.document_token_list.append(token_list)
                self.document_size_array[i] = len(token_list)
                self.count_document_token_stats(token_list)
                i += 1
            print("    Finished {} documents".format(i))
        f.close()
        f_out.close()

    def count_document_token_stats(self, token_list):
        for token in token_list:
            if token not in self.type_freq_dict:
                self.type_freq_dict[token] = 1
                self.type_list.append(token)
                self.type_index_dict[token] = self.num_types
                self.num_types += 1
            else:
                self.type_freq_dict[token] += 1
            self.num_tokens += 1

    def get_cleaned_token_list(self, line):
        # if we wanted to do preprocessing, we would do it here
        token_list = (line.strip().strip('\n').strip()).split(" ")
        return token_list

    def sort_types(self):
        print("    Sorting type list by method {}".format(self.sort_types_method))

        if self.sort_types_method == 'freq':
            sorted_types = sorted(self.type_freq_dict.items(), key=operator.itemgetter(1), reverse=True)
        elif self.sort_types_method == 'alphabetical':
            sorted_types = sorted(self.type_freq_dict.items(), key=operator.itemgetter(0))
            # given a list of tuples, and a number telling you which spot in each tuple you care about, sort by that spot
        elif self.sort_types_method is None:
            pass
        else:
            raise AttributeError('Invalid arg {} to sort_types_method'.format(self.sort_types_method))

        self.type_list = []
        self.type_index_dict = {}
        for i in range(len(sorted_types)):
            current_type = sorted_types[i][0]
            self.type_list.append(current_type)
            self.type_index_dict[current_type] = i

    def save_corpus(self):
        print("    Saving corpus")

        pickle_file = open(self.corpus_path + '/corpus_object.p', 'wb')
        pickle.dump(self, pickle_file)
        pickle_file.close()

        f = open(self.corpus_path + '/config.csv', 'w')
        f.write('corpus_name: {}\n'.format(self.corpus_name))
        f.write('num_documents: {}\n'.format(self.num_documents))
        f.write('num_types: {}\n'.format(self.num_types))
        f.write('num_tokens: {}\n'.format(self.num_tokens))
        f.close()

        f = open(self.corpus_path + '/documents.csv', 'w')
        for i in range(self.num_documents):
            doc_name = self.document_name_list[i]
            doc_index = self.document_index_dict[doc_name]
            doc_size = self.document_size_array[i]
            f.write("{},{},{}\n".format(doc_index, doc_name, doc_size))
        f.close()

        f = open(self.corpus_path + '/types.csv', 'w')
        for i in range(self.num_types):
            current_type = self.type_list[i]
            type_index = self.type_index_dict[current_type]
            type_freq = self.type_freq_dict[current_type]
            f.write("{},{},{}\n".format(type_index, current_type, type_freq))
        f.close()

    def load_corpus(self, corpus_name):
        pickle_file = open('../corpora/'+corpus_name+'/corpus_object.p', 'rb')
        instance_object = pickle.load(pickle_file)
        pickle_file.close()
        self.__dict__ = instance_object.__dict__
