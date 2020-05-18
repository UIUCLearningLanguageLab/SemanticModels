from datasets import simple_feature_categories
from models import prototype_classifier
from datasets import mcrae_feature_norms
import numpy as np


def main():
    np.set_printoptions(precision=3)

    # sfc_dataset = simple_feature_categories.SimpleFeatureCategories()
    # sfc_dataset.create_dataset()
    # sfc_dataset.create_training_folds()
    # prototype_model = prototype_classifier.PrototypeClassifier(sfc_dataset, verbose=True)

    mcrae_dataset = mcrae_feature_norms.McRaeFeatureNorms()
    mcrae_dataset.create_dataset('../datasets/mcrae_feature_norms', categories=True)

main()