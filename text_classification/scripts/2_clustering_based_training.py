import os

cluster_number = int(os.environ['cluster_number'])
random_state = int(os.environ['random_state'])
base_parth = os.environ['base_path']
model_path = os.environ['models_path']
data_path = os.environ['data_path']
check_point_dir = f"./models_cluster_{os.environ['cluster_number']}"


data_path_test = os.environ['s2_data_path_test']
data_path_train = os.environ['s2_data_path_train']
model_name = os.environ['s2_model_name']
batch_size = int(os.environ['s2_batch_size'])
epochs = int(os.environ['s2_epochs'])
lr = float(os.environ['s2_lr'])
save_dir_model_base = os.environ['s2_save_dir_model_base']

save_finetuned_predictions = f'{data_path}/model_cluster_{cluster_number}_prediction.pkl'
save_finetuned_metrics = f'{data_path}/metrics_cluster_{cluster_number}.pkl'
save_model_path = f'{model_path}/finetuned_cluster_{cluster_number}'



from datasets import load_dataset
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import torch

random.seed(random_state)
tf.random.set_seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)

df_train = pd.read_pickle(data_path_train)
df_val = pd.read_pickle(f"{data_path_test[:-4]}.pkl")
df_test = pd.read_pickle(data_path_test)

df_train = df_train[df_train.cluster_labels == cluster_number].reset_index()
df_val = df_val[df_val.cluster_labels == cluster_number].reset_index()
df_test = df_test[df_test.cluster_labels == cluster_number].reset_index()


df_train.columns = list(['base_idx'] + list(df_train.columns[1:]))
df_val.columns = list(['base_idx'] + list(df_train.columns[1:]))
df_test.columns = list(['base_idx'] + list(df_test.columns[1:]))

df_train.head(3)

"""# Fine tuning the model over cluster dataset

"""


import ktrain
from ktrain import text as ktrain_text

transformer_model = ktrain_text.Transformer(model_name, class_names=df_train['label'].unique().tolist())
transformer_model.get_config()
trn = transformer_model.preprocess_train(df_train['text'].values, df_train['label'].values, verbose=False)
model = transformer_model.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, batch_size=batch_size)

learner.fit(lr,epochs,checkpoint_folder=check_point_dir)

"""## Validation metrics

---
"""


from sklearn.metrics import classification_report,confusion_matrix
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn import metrics
import seaborn as sns
import os

def validation_metrics(y_pred, y_true, show_img=True):
  print('Confusion Matrix')
  matrix_confusion = confusion_matrix(y_true, y_pred)
  print(matrix_confusion)

  if show_img:
    sns.heatmap(matrix_confusion/matrix_confusion.sum(axis=1)[:,None],
                annot=True, vmin=0.0, vmax=1.0)
    plt.show()

  dict_aux = {
      'accuracy' :[ metrics.accuracy_score(y_true, y_pred)],
      'f1-macro' : [metrics.f1_score(y_true, y_pred, average='macro')],
      'f1-micro' : [metrics.f1_score(y_true, y_pred, average='micro')],
      'recall macro'    : [metrics.recall_score(y_true, y_pred, average='macro')],
      'recall micro'    : [metrics.recall_score(y_true, y_pred, average='micro')],
      'precision macro' : [metrics.precision_score(y_true, y_pred, average='macro')],
      'precision micro' : [metrics.precision_score(y_true, y_pred, average='micro')]

  }
  return pd.DataFrame.from_dict(dict_aux)

def check_epoch_performance(cur_df ,weights_path):
  model.load_weights(weights_path)
  predictor = ktrain.get_predictor(model, transformer_model)

  x_test = cur_df['text'].values
  y_hat = predictor.predict_proba(x_test)
  y_hat = np.argmax(y_hat, axis=1)

  y_true = cur_df['label'].values

  print("+", "=" * 50, "+")
  print("\n\n")

  metrics = validation_metrics(y_hat, y_true)
  display(metrics)

  print("+", "=" * 50, "+")
  print("\n\n")

  return metrics['f1-macro'].iloc[0], predictor, metrics, y_hat

#########################
## Get predictions val ##
#########################
print("########################")
print("## Validation Metrics ##")
print("########################")

val_best_f1 = -1
val_best_predictor = None
val_best_metrics = None
val_best_epoch = None
val_best_pred = None

for cur_epoch in range(epochs):
  cur_f1, cur_predictor, cur_metrics, cur_y_hat = check_epoch_performance(df_val, f"{check_point_dir}/weights-0{cur_epoch + 1}.hdf5")
  if val_best_f1 < cur_f1 :
    val_best_f1 = cur_f1
    val_best_predictor = cur_predictor
    val_best_metrics = cur_metrics
    val_best_epoch = cur_epoch
    val_best_metrics['epoch'] = cur_epoch
    val_best_pred = cur_y_hat

print(f"## best val epoch {val_best_epoch}")
val_best_predictor.save(save_model_path)
df_val['prediction'] = val_best_pred
df_val.to_pickle(f'{save_finetuned_predictions[:-4]}_val.pkl')
val_best_metrics.to_pickle(f'{save_finetuned_metrics[:-4]}_val.pkl')

##########################
## Get predictions test ##
##########################
print("########################")
print("##### Test Metrics #####")
print("########################")

best_f1 = -1
best_metrics = None
best_epoch = None
best_pred = None

for cur_epoch in range(epochs):
  cur_f1, _, cur_metrics, cur_y_hat = check_epoch_performance(df_test, f"{check_point_dir}/weights-0{cur_epoch + 1}.hdf5")
  if best_f1 < cur_f1 :
    best_f1 = cur_f1

    best_metrics = cur_metrics
    best_epoch = cur_epoch
    best_metrics['epoch'] = cur_epoch
    best_pred = cur_y_hat

print(f"## best test epoch {best_epoch}")
df_test['prediction'] = best_pred
df_test.to_pickle(save_finetuned_predictions)
best_metrics.to_pickle(save_finetuned_metrics)