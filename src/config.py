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