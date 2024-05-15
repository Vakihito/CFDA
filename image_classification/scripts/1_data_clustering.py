import os


random_state = int(os.environ['random_state'])
base_parth = os.environ['base_path']
models_path = os.environ['models_path']
data_path = os.environ['data_path']
n_clusters = int(os.environ['n_clusters'])
kmeans_model_save_path = os.environ['s1_kmeans_model_save_path']
s1_data_path = os.environ['s1_data_path']

dataset_name = os.environ['dataset_name']
val_dataset_name = os.environ['val_dataset_name']
output_data_path_train = os.environ['output_data_path_train']
output_data_path_test = os.environ['output_data_path_test']

from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_pickle(s1_data_path)

df_train = df[~df['dataset'].isin([dataset_name, val_dataset_name])].reset_index(drop=True)
df_val = df[df['dataset'] == val_dataset_name].reset_index(drop=True)
df_test = df[df['dataset'] == dataset_name].reset_index(drop=True)

k_means = KMeans(n_clusters=n_clusters, random_state=random_state).fit(df_train['encoded_texts'].values.tolist())

df_train['cluster_labels'] = k_means.labels_

df_train.to_pickle(output_data_path_train)

import pickle

with open(kmeans_model_save_path, 'wb') as f:
    pickle.dump(k_means, f)

df_test['cluster_labels'] = k_means.predict(df_test['encoded_texts'].values.tolist())
df_val['cluster_labels'] = k_means.predict(df_val['encoded_texts'].values.tolist())

df_test.to_pickle(output_data_path_test)
df_val.to_pickle(f"{output_data_path_test[:-4]}_val.pkl") 