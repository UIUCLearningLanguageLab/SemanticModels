from datasets import simple_feature_categories
from models import prototype_classifier
from datasets import mcrae_feature_norms
from models import logistic_regression_classifier
from models import multilayer_classifier
from corpora import corpus
import numpy as np


def main():
    np.set_printoptions(precision=3)

    childes_10d = corpus.Corpus()
    childes_1d = corpus.Corpus()
    childes_100d = corpus.Corpus()
    childes = corpus.Corpus()

    childes_1d.create_corpus_object("childes_1d", "../external_datasets/childes_1d")
    childes_100d.create_corpus_object("childes_100d", "../external_datasets/childes_100d")
    childes.create_corpus_object("childes", "../external_datasets/childes")
    #childes_10d.create_corpus_object("childes_10d", "../external_datasets/childes_10d")
    #childes_10d.load_corpus("childes_10d")

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