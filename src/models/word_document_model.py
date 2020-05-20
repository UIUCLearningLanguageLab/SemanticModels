from models import distributional_model
import config
import numpy as np


class WordDocumentEmbedding(distributional_model.DistributionalModel):
    def __init__(self):
        super().__init__()

    def create_model(self, corpus,
                     num_vocab=config.WordDocumentEmbedding.num_vocab,
                     stop_list_path=config.WordDocumentEmbedding.stop_list_path,
                     normalization_method=config.WordDocumentEmbedding.normalization_method,
                     reduction_method=config.WordDocumentEmbedding.reduction_method,
                     reduction_size=config.WordDocumentEmbedding.reduction_size):

        self.corpus = corpus
        self.stop_list_path = stop_list_path
        self.normalization_method = normalization_method
        self.reduction_method = reduction_method
        self.reduction_size = reduction_size
        # num vocab

        # in the DM base class
        self.create_model_name("wd")
        self.create_model_directory()
        self.create_stop_list()
        self.create_vocabulary(num_vocab)
        self.create_model_config_files()

        # LSA specific
        self.add_word_document_embedding_config_info()
        self.create_word_document_embedding()

        # in the DM base class
        self.normalize_embedding_matrix()
        self.reduce_embedding_matrix()
        self.save_model()
        self.save_embedding_matrix()

    def add_word_document_embedding_config_info(self):
        f = open(self.model_path+'/config.txt', 'a')
        f.write("normalization_method: {}\n".format(self.normalization_method))
        f.write("reduction_method: {}\n".format(self.reduction_method))
        f.write("reduction_size: {}\n".format(self.reduction_size))
        f.close()
    
    def create_word_document_embedding(self):
        print("    Processing Corpus")
        self.vocab_embedding_matrix = np.zeros([self.num_vocab, self.corpus.num_documents])
        for i in range(self.corpus.num_documents):
            current_token_list = self.corpus.document_token_list[i]
            for token in current_token_list:
                if token in self.vocab_index_dict:
                    self.vocab_embedding_matrix[self.vocab_index_dict[token], i] += 1
                else:
                    self.vocab_embedding_matrix[self.vocab_index_dict["UNKNOWN"], i] += 1
            if i % 100 == 0:
                print("        Finished {} documents".format(i))
        