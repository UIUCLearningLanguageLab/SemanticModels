# from datasets import simple_feature_categories
# from models import prototype_classifier
# from datasets import mcrae_feature_norms
# from models import logistic_regression_classifier
# from models import multilayer_classifier
from corpora import corpus
from models import word_document_model
import numpy as np


def main():
    np.set_printoptions(precision=3)

    childes_10d = corpus.Corpus()
    childes_1d = corpus.Corpus()
    childes_100d = corpus.Corpus()
    childes = corpus.Corpus()
    childes_1d.create_corpus_object("childes_1d", "../external_datasets/childes_1d")
    childes_10d.create_corpus_object("childes_10d", "../external_datasets/childes_10d")
    childes_100d.create_corpus_object("childes_100d", "../external_datasets/childes_100d")
    childes.create_corpus_object("childes", "../external_datasets/childes")
    # childes_1d.load_corpus("childes_1d")
    # childes_10d.load_corpus("childes_10d")
    # childes_100d.load_corpus("childes_100d")
    # childes.load_corpus("childes")

    wd_model = word_document_model.WordDocumentEmbedding()
    wd_model.create_model(childes)
    #wd_model.load_model("wd_5_19_12_45_30_1")
    wd_model.compute_full_similarity_matrix('cosine')
    wd_model.get_nearest_neighbors(20)

    # # sfc_dataset = simple_feature_categories.SimpleFeatureCategories()
    # # sfc_dataset.create_dataset()
    # # sfc_dataset.create_training_folds()
    # # prototype_model = prototype_classifier.PrototypeClassifier(sfc_dataset, verbose=True)

    # mcrae_dataset = mcrae_feature_norms.McRaeFeatureNorms()
    # #mcrae_dataset.create_dataset(categories=True)
    # mcrae_dataset.load_dataset('mcrae_feature_norms')
    # mcrae_dataset.create_training_folds()
    # prototype_model = prototype_classifier.PrototypeClassifier(mcrae_dataset, verbose=False)
    # logistic_regression_model = logistic_regression_classifier.LogisticRegressionClassifier(mcrae_dataset, verbose=False)
    # multilayer_model = multilayer_classifier.NumpyMultilayerClassifier(mcrae_dataset, verbose=False)

main()