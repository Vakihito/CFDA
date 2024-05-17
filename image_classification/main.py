from dotenv import dotenv_values
import os

# exporting the enviroment variables

env_vars = dotenv_values(f"question_answering/.env")
for key, value in env_vars.items():
    os.environ[key] = value

base_path = os.environ["main_path"]
n_clusters = int(os.environ["n_clusters"])


os.system(f"python3 {base_path}/scripts/1_data_split.py")

os.system(f"python3 {base_path}/scripts/2_image_classification_simple_fine_tuning.py")

os.system(f"python3 {base_path}/scripts/2_5_embedding_extraction_and_cluster.py.py")

for cur_cluster in range(int(os.environ["n_clusters"])):
    os.environ['cur_cluster'] = str(cur_cluster)
    os.system(f"python3 {base_path}/scripts/3_image_classification_cluster_fine_tuning.py")

os.system(f"python3 {base_path}/scripts/4_0_ensamble_prediction.py")

os.system(f"python3 {base_path}/scripts/5_cst.py")

os.system(f"python3 {base_path}/scripts/6_get_cft_prediction.py")