print("###############################")
print("##### Starting notebook 4_0 #####")
print("###############################")
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from IPython.display import display
from sklearn.cluster import KMeans
import pandas as pd
import os
import torch.nn as nn
import random
import numpy as np
import torch

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

# reading envars #
save_path = os.environ['save_path']
n_clusters = int(os.environ["n_clusters"])
###

df_test = pd.read_pickle(f"{save_path}/test_data_clustered.pkl")
df_test = df_test[["dataset", "index","image_name",	"data_path",	"embeddings",	"cluster_labels"]].reset_index(drop=True)
df_train = pd.read_pickle(f"{save_path}/train_data_clustered.pkl")
inicial_columns = list(df_train.columns)
df_simple = pd.read_pickle(f'{save_path}/simple_predictions.pkl')

# loading the cluster predictions #
list_dfs_cluster = []
for cur_cluster in range(n_clusters):
  cur_df = pd.read_pickle(f'{save_path}/cluster_{cur_cluster}_predictions.pkl')
  list_dfs_cluster.append(cur_df)
df_cluster = pd.concat(list_dfs_cluster)
###

confidence_cluster = np.array(list(df_cluster['confidence'].values))
confidence_simple = np.array(list(df_simple['confidence'].values))

def get_mapping_data(cur_df):
  map_idx_pred = {}
  for idx, row in cur_df.iterrows():
    map_idx_pred[row['index']] = row['confidence']
  return map_idx_pred

map_simple_confidence = get_mapping_data(df_simple)
map_cluster_confidence = get_mapping_data(df_cluster)

df_test["simple_confidence"] = df_test["index"].apply(lambda x: map_simple_confidence[x])
df_test["cluster_confidence"] = df_test["index"].apply(lambda x: map_cluster_confidence[x])

simple_confidence = np.array(df_test["simple_confidence"].values.tolist())
cluster_confidence = np.array(df_test["cluster_confidence"].values.tolist())
df_test["confidence"] = list((simple_confidence + cluster_confidence) / 2)
df_test["max_confidence"] = df_test["confidence"].apply(lambda x: np.max(x))
df_test["avg_label"] = df_test["confidence"].apply(lambda x: np.argmax(x))

df_test.head(3)

df_cut = df_test[df_test["max_confidence"] > 0.6].reset_index(drop=True)
df_cut["sentiment"] = df_cut["avg_label"].values

df_train = pd.concat([df_cut, df_train])
df_train = df_train[inicial_columns].reset_index(drop=True)

df_train.to_pickle(f"{save_path}/train_data_clustered_ensamble.pkl")