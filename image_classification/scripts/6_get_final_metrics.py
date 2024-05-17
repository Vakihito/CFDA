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

save_path = os.environ['save_path']
images_path = os.environ["read_data_path"]
test_dataset = os.environ["dataset_test"]
n_clusters = int(os.environ["n_clusters"])
output_model_path = os.environ['output_model_path']

df_test = pd.read_pickle(f"{save_path}/test_data_clustered.pkl")
df_simple = pd.read_pickle(f'{save_path}/simple_predictions.pkl')

list_dfs_cluster = []
for cur_cluster in range(n_clusters):
  cur_df = pd.read_pickle(f'{save_path}/cluster_{cur_cluster}_predictions.pkl')
  list_dfs_cluster.append(cur_df)

df_cluster = pd.concat(list_dfs_cluster)

def add_prediction_to_data(cur_df, prediction_type, df_target):
  l_index = cur_df['index'].values
  l_predictions = cur_df['prediction'].values
  map_index_to_prediction = {i : j for i, j in zip(l_index, l_predictions)}
  df_target[prediction_type] = df_target['index'].apply(lambda x: map_index_to_prediction[x])
  return df_target

df_test = add_prediction_to_data(df_simple, "simple", df_test)
df_test = add_prediction_to_data(df_cluster, "cluster", df_test)

from sklearn.metrics import classification_report,confusion_matrix
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn import metrics
import seaborn as sns
import os

def validation_metrics(y_true, y_pred, metric_type):
  dict_aux = {
      'accuracy' :[ metrics.accuracy_score(y_true, y_pred)],
      'f1-macro' : [metrics.f1_score(y_true, y_pred, average='macro')],
      'f1-micro' : [metrics.f1_score(y_true, y_pred, average='micro')],
      'recall macro'    : [metrics.recall_score(y_true, y_pred, average='macro')],
      'recall micro'    : [metrics.recall_score(y_true, y_pred, average='micro')],
      'precision macro' : [metrics.precision_score(y_true, y_pred, average='macro')],
      'precision micro' : [metrics.precision_score(y_true, y_pred, average='micro')]

  }
  metrics_df = pd.DataFrame(dict_aux)
  metrics_df["type"] = metric_type
  return metrics_df

df_test.head(3)

all_metrics = []
all_metrics.append(validation_metrics(df_test["sentiment"].values, df_test["simple"].values, "simple"))
all_metrics.append(validation_metrics(df_test["sentiment"].values, df_test["cluster"].values, "cluster"))
metrics_cst = pd.read_pickle(f'{save_path}/cluster_fusion_predictions_cluster_att.pkl')
all_metrics.append(metrics_cst)

all_results = pd.concat(all_metrics)
display(all_results)
all_results.to_pickle(f"{save_path}/results.pkl")