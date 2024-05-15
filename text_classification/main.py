from dotenv import dotenv_values
import os

# exporting the enviroment variables
env_vars = dotenv_values(".env")
for key, value in env_vars.items():
    os.environ[key] = value
    
    
os.system("python3 image_classification/scripts/0_train_model_over_dataset.py")

os.system("python3 image_classification/scripts/1_data_clustering.py")

os.system("python3 image_classification/scripts/2_clustering_based_training.py")

os.system("python3 image_classification/scripts/3_compare_models.py")

os.system("python3 image_classification/scripts/4_get_embeddings_cluster_models.py")

os.system("python3 image_classification/scripts/5_cst.py")
