from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import random
import numpy as np


np.random.seed(0)
random.seed(0)

read_data_path = os.environ["read_data_path"]
n_clustes = int(os.environ["n_clusters"])

data_path = os.environ["s1_data_path"]
test_dataset = os.environ["dataset_test"]
val_dataset =  os.environ["dataset_test"]
random_state = 0

save_path = os.environ['save_path']

df = pd.read_pickle(data_path)

df_train = df[(df['dataset'] != test_dataset)].reset_index(drop=True)
df_test = df[(df['dataset'] == test_dataset)].reset_index(drop=True)

df_train = df_train.reset_index(drop=True)
df_val = df_test.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

display(df_train.head(3))
display(df_val.head(3))
display(df_test.head(3))

# cluseteing #
# k_means = KMeans(n_clusters=n_clustes, random_state=random_state).fit(df_train['embeddings'].values.tolist())

# ## getting the cluster of each dataset
# df_train['cluster_labels'] = k_means.labels_
# df_test['cluster_labels'] = k_means.predict(df_test['embeddings'].values.tolist())
# df_val['cluster_labels'] = k_means.predict(df_val['embeddings'].values.tolist())

# print("train dataframe distribution : ")
# print(df_train['cluster_labels'].value_counts())

# print("test dataframe distribution : ")
# print(df_test['cluster_labels'].value_counts())
###

df_train.to_pickle(f"{save_path}/train_data.pkl")
df_val.to_pickle(f"{save_path}/val_data.pkl")
df_test.to_pickle(f"{save_path}/test_data.pkl")