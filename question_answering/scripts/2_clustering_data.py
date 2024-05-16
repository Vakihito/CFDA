from sklearn.cluster import KMeans
import pandas as pd
import os
from sklearn.model_selection import train_test_split

read_data_path = os.environ["read_data_path"]
n_clustes = int(os.environ["n_clusters"])

data_path = os.environ["1_data_path"]
test_dataset = os.environ["dataset_test"]
val_dataset =  os.environ["dataset_val"]
random_state = 0

df = pd.read_pickle(data_path)

df_train = df[~df['dataset'].isin([test_dataset, val_dataset]) ].reset_index(drop=True)
df_test = df[df['dataset'] == test_dataset].reset_index(drop=True)
df_val = df[df['dataset'] == val_dataset].reset_index(drop=True)

df_train =  df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

df_train.head(3)

k_means = KMeans(n_clusters=n_clustes, random_state=random_state).fit(df_train['context_and_question_encoded'].values.tolist())

## getting the cluster of each dataset
df_train['cluster_labels'] = k_means.labels_
df_test['cluster_labels'] = k_means.predict(df_test['context_and_question_encoded'].values.tolist())
df_val['cluster_labels'] = k_means.predict(df_val['context_and_question_encoded'].values.tolist())

print("train dataframe distribution : ")
print(df_train['cluster_labels'].value_counts())

print("test dataframe distribution : ")
print(df_test['cluster_labels'].value_counts())

df_train.to_pickle(f"{read_data_path}/train_data_clustered.pkl")
df_val.to_pickle(f"{read_data_path}/val_data_clustered.pkl")
df_test.to_pickle(f"{read_data_path}/test_data_clustered.pkl")

