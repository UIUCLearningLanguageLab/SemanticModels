from datasets import simple_feature_categories

def main():
    #sfc_dataset_name = 'sfc_6_300_10_5_16_19_58_17_5'
    sfc_dataset = simple_feature_categories.SimpleFeatureCategories()
    sfc_dataset.create_dataset()
    #sfc_dataset.load_dataset(sfc_dataset_name)
    sfc_dataset.create_training_folds()



main()