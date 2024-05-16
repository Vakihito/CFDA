from dotenv import dotenv_values
import os

# exporting the enviroment variables

env_vars = dotenv_values(f"question_answering/.env")
for key, value in env_vars.items():
    os.environ[key] = value

base_path = os.environ["main_path"]
n_clusters = int(os.environ["n_clusters"])


os.system(f"python3 {base_path}/scripts/1_clustering_data.py")

os.system(f"python3 {base_path}/scripts/2_simple_finetuning.py")

for cur_cluster in range(int(n_clusters)):
    os.environ['cluster_subset'] = str(cur_cluster)
    os.system(f"python3 {base_path}/scripts/3_cluster_base_finetuning_cluster.py")

os.system(f"python3 {base_path}/scripts/4_get_ensamble_predictions.py")

os.system(f"python3 {base_path}/scripts/5_cst.py")

os.system(f"python3 {base_path}/scripts/6_get_cft_prediction.py")