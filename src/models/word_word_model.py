from models import distributional_model
import config
import numpy as np
from cytoolz import itertoolz


class WordWordEmbedding(distributional_model.DistributionalModel):
    #############################################################################################################################
    def __init__(self):
        super().__init__()

        

    #############################################################################################################################
    def create_model(self, corpus,
                     num_vocab=config.WordWordEmbedding.num_vocab,
                     stop_list_path=config.WordWordEmbedding.stop_list_path,
                     window_size=config.WordWordEmbedding.window_size,
                     window_type=config.WordWordEmbedding.window_type,
                     window_weight=config.WordWordEmbedding.window_weight,
                     normalization_method=config.WordWordEmbedding.normalization_method,
                     reduction_method=config.WordWordEmbedding.reduction_method,
                     reduction_size=config.WordWordEmbedding.reduction_size):

        print("Creating Word-Word Embeddings from {} with vocab size = {}".format(corpus.corpus_name, num_vocab))
        self.corpus = corpus

        self.stop_list_path = stop_list_path
        self.window_size = window_size
        self.window_type = window_type
        self.window_weight = window_weight

        self.normalization_method = normalization_method
        self.reduction_method = reduction_method
        self.reduction_size = reduction_size

        self.create_model_name("ww")
        self.create_model_directory()
        self.create_stop_list()
        self.create_vocabulary(num_vocab)

        self.create_model_config_files()
        self.add_word_word_embedding_config_info()
        self.create_word_word_embedding()
        self.normalize_embedding_matrix()

        self.reduce_embedding_matrix()
        self.save_model()
        self.save_embedding_matrix()
    
    def add_word_word_embedding_config_info(self):
        f = open(self.model_path+'/config.txt', 'a')
        f.write("stop_list_path: {}\n".format(self.stop_list_path))
        f.write("window_size: {}\n".format(self.window_size))
        f.write("window_type: {}\n".format(self.window_type))
        f.write("window_weight: {}\n".format(self.window_weight))
        f.write("normalization_method: {}\n".format(self.normalization_method))
        f.write("reduction_method: {}\n".format(self.reduction_method))
        f.write("reduction_size: {}\n".format(self.reduction_size))
        f.close()
    
    def create_word_word_embedding(self):
        print("    Counting word-word co-occurrences in {}-word moving window".format(self.window_size))
        count_matrix = np.zeros([self.num_vocab, self.num_vocab])

        for i in range(self.corpus.num_documents):
            current_token_list = self.corpus.document_token_list[i] + ['*PAD*'] * self.window_size
            windows = itertoolz.sliding_window(self.window_size + 1, current_token_list)
            #  [(1,2,3,4), (2,3,4,5), ...]
            # flat 1 1 1
            # lin  3 2 1
            # nlin 4 2 1
            counter = 0
            for w in windows:
                for t1, t2, dist in zip([w[0]] * self.window_size, w[1:], range(self.window_size)):
                    # [1, 1, 1], [2, 3, 4], [0, 1, 2] ---> [(1,2,0), (1,3,1), (1,4,2)]
                    if t1 == '*PAD*' or t2 == '*PAD*':
                        continue
                    if t1 not in self.vocab_index_dict:
                        t1 = "UNKNOWN"
                    if t2 not in self.vocab_index_dict:
                        t2 = "UNKNOWN"
                    t1_id = self.vocab_index_dict[t1]
                    t2_id = self.vocab_index_dict[t2]

                    if self.window_weight == "linear":
                        count_matrix[t1_id, t2_id] += self.window_size - dist
                    elif self.window_weight == "flat":
                        count_matrix[t1_id, t2_id] += 1
                    else:
                        raise AttributeError('Invalid arg to "window_weight".')
                counter += 1
        
        # window_type
        if self.window_type == 'forward':
            self.vocab_embedding_matrix = count_matrix
        elif self.window_type == 'backward':
            self.vocab_embedding_matrix = count_matrix.transpose()
        elif self.window_type == 'summed':
            self.vocab_embedding_matrix = count_matrix + count_matrix.transpose()
        else:
            raise AttributeError('Invalid arg to "window_type".')
