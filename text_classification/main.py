from dotenv import dotenv_values
import os

# exporting the enviroment variables

env_vars = dotenv_values(f"text_classification/.env")
for key, value in env_vars.items():
    os.environ[key] = value

base_path = os.environ["base_path"]
n_clusters = int(os.environ["n_clusters"])


os.system(f"python3 {base_path}/scripts/0_train_model_over_dataset.py")

os.system(f"python3 {base_path}/scripts/1_data_clustering.py")

for cur_cluster in range(int(n_clusters)):
    os.environ['cluster_number'] = str(cur_cluster)
    os.system(f"python3 {base_path}/scripts/2_clustering_based_training.py")

os.system(f"python3 {base_path}/scripts/3_compare_models.py")

for i in range(int (n_clusters)):
    os.environ["cur_cluster_number"] = str(i)
    os.system(f"python3 {base_path}/scripts/4_get_embeddings_cluster_models.py")

os.system(f"python3 {base_path}/scripts/5_cst.py")
