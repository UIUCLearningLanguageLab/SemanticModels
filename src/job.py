from datasets import simple_feature_categories
from models import prototype_classifier
import numpy as np


def main():
    np.set_printoptions(precision=3)

    sfc_dataset = simple_feature_categories.SimpleFeatureCategories()
    sfc_dataset.create_dataset()
    sfc_dataset.create_training_folds()
    prototype_model = prototype_classifier.PrototypeClassifier(sfc_dataset, verbose=True)

main()