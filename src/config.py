class InstanceFeatureDataset:
    num_folds = 5

class SimpleFeatureCategories:
    num_categories = 6
    category_size_list = 50  # if a single value, all categories same size, otherwise make it a list
    num_features = 10
    global_mean = 0
    global_stdev = 1
    within_stdev = 1
    num_folds = 2

class PrototypeClassifier:
    similarity_metric = 'cosine'

class LogisticRegressionClassifier:
    learning_rate = 0.10
    weight_init_stdev = 0.01
    num_epochs = 200
    output_freq = 10
    save_f1_history = True
    save_ba_history = True

class MultilayerClassifier:
    hidden_size = 10
    learning_rate = 0.10
    weight_init_stdev = 0.01
    num_epochs = 200
    hidden_activation_function = 'sigmoid'   # can be 'sigmoid' or 'tanh'
    output_freq = 10
    save_f1_history = True
    save_ba_history = True

class Corpus:
    sort_types_method = 'freq'    # can be 'freq' or 'alphabetical' or None

class WordDocumentEmbedding:
    num_vocab = 4095   # UNKNOWN
    stop_list_path = None
    normalization_method = 'ppmi'
    reduction_method = 'svd'
    reduction_size = 50