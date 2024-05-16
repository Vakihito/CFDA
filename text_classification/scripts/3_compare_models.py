print("###############################")
print("##### Starting notebook 3 #####")
print("###############################")

import os

base_parth = os.environ['base_path']
models_path = os.environ['models_path']
data_path = os.environ['data_path']
n_clusters = int(os.environ['n_clusters'])

save_predictions_all_data = os.environ['s3_save_predictions_all_data']
save_metrics_comp = os.environ['s3_save_metrics_comp']

import pandas as pd
from sklearn import metrics

def validation_metrics(y_pred, y_true, model_type, show_img=False):

  dict_aux = {
      'accuracy' :[ metrics.accuracy_score(y_true, y_pred)],
      'f1-macro' : [metrics.f1_score(y_true, y_pred, average='macro')],
      'f1-micro' : [metrics.f1_score(y_true, y_pred, average='micro')],
      'recall macro'    : [metrics.recall_score(y_true, y_pred, average='macro')],
      'recall micro'    : [metrics.recall_score(y_true, y_pred, average='micro')],
      'precision macro' : [metrics.precision_score(y_true, y_pred, average='macro')],
      'precision micro' : [metrics.precision_score(y_true, y_pred, average='micro')],
      'model_type' : [model_type]
  }
  return pd.DataFrame.from_dict(dict_aux)

list_all_cluster_preds = []
for i in range(n_clusters):
  if (os.path.exists(f'{data_path}/model_cluster_{i}_prediction.pkl')):
    list_all_cluster_preds.append(pd.read_pickle(f'{data_path}/model_cluster_{i}_prediction.pkl'))

df_cluster_preds = pd.concat(list_all_cluster_preds).sort_values(by='text').reset_index(drop=True)
df_all_preds = pd.read_pickle(save_predictions_all_data).sort_values(by='text').reset_index(drop=True)

assert all([i == j for i, j in zip(df_cluster_preds['idx'].values, df_all_preds['idx'].values)])
assert all([i == j for i, j in zip(df_cluster_preds['label'].values, df_all_preds['label'].values)])
assert all([i == j for i, j in zip(df_cluster_preds['text'].values, df_all_preds['text'].values)])

y_true = df_all_preds['label'].values

y_pred_cluster = df_cluster_preds['prediction'].values
y_pred_all = df_all_preds['prediction'].values

df_cluster_pred = validation_metrics(y_pred_cluster, y_true, 'cluster_finetuning')
df_all_pred = validation_metrics(y_pred_all, y_true, 'finetuning_all')

pd.concat([df_all_pred, df_cluster_pred]).to_pickle(save_metrics_comp)