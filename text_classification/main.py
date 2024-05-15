from dotenv import dotenv_values
import os

# exporting the enviroment variables
env_vars = dotenv_values(".env")
for key, value in env_vars.items():
    os.environ[key] = value
    
base_path = os.environ['base_path']

os.system(f"python3 {base_path}/scripts/0_train_model_over_dataset.py")

os.system(f"python3 {base_path}/scripts/1_data_clustering.py")

os.system(f"python3 {base_path}/scripts/2_clustering_based_training.py")

os.system(f"python3 {base_path}/scripts/3_compare_models.py")

os.system(f"python3 {base_path}/scripts/4_get_embeddings_cluster_models.py")

os.system(f"python3 {base_path}/scripts/5_cst.py")
