import os

random_state = int(os.environ['random_state'])
dataset_name = os.environ['dataset_name']
val_dataset_name = os.environ['val_dataset_name']

base_path = os.environ['base_path']
models_path = os.environ['models_path']
data_path = os.environ['data_path']
random_state = int(os.environ['random_state'])
check_point_dir = './models'
save_dir_model = os.environ['s0_save_dir_model']
model_name = os.environ['s0_model_name']
batch_size = int(os.environ['s0_batch_size'])
epochs = int(os.environ['s0_epochs'])
lr = float(os.environ['s0_lr'])

save_predictions_all_data = os.environ['s0_save_predictions_all_data']
save_metrics = os.environ['s0_save_metrics']
data_path = os.environ['s0_data']

import random
import pandas as pd
import numpy as np
import tensorflow as tf
import torch

random.seed(random_state)
tf.random.set_seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)

df = pd.read_pickle(data_path)

df['dataset'].unique()

df_train = df[~df['dataset'].isin([dataset_name, val_dataset_name])].reset_index(drop=True)
df_val = df[df['dataset'] == val_dataset_name].reset_index(drop=True)
df_test = df[df['dataset'] == dataset_name].reset_index(drop=True)


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
val_best_predictor.save(save_dir_model)
df_val['prediction'] = val_best_pred
df_val.to_pickle(f'{save_predictions_all_data[:-4]}_val.pkl')
val_best_metrics.to_pickle(f'{save_metrics[:-4]}_val.pkl')

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
df_test.to_pickle(save_predictions_all_data)
best_metrics.to_pickle(save_metrics)
