import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import seaborn as sns
from torch import optim
from random import randint
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix

import seaborn as sn
from sklearn import metrics
import pandas as pd
import os
import torch
import tensorflow as tf


number_clusters = int(os.environ["n_clusters"])
number_of_training_epochs = int(os.environ['s5_number_of_training_epochs'] )
number_of_layers = int(os.environ['s5_number_of_layers'] )
learning_rate = float(os.environ['s5_learning_rate'] )
random_state = int(os.environ['random_state'])

base_path = os.environ['base_path']
model_path = os.environ['models_path']
data_path = os.environ['data_path']

test_path = os.environ['s5_data_path_test']
train_path = os.environ['s5_data_path_train']

results_path = os.environ['s5_results']
save_predictions = os.environ['s5_predictions']

random.seed(random_state)
tf.random.set_seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)

import pandas as pd
df_test = pd.read_pickle(test_path)
df_train = pd.read_pickle(train_path)


number_labels = len(df_train['label'].unique())
n_clusters = len(df_train['cluster_labels'].unique())

class EmbeddingsDataset(Dataset):
  def __init__(self, cur_df, number_clusters):
    self.X = np.array(cur_df[[f'embeddings_cluster_{i}' for i in range(number_clusters)]].values.tolist())
    self.Y = torch.from_numpy(cur_df["label"].values)

    ## define numero de samples de cada split
    self.n_samples = len(self.Y)

  def __getitem__(self, index):
    return self.X[index], self.Y[index]

  def __len__(self):
    return self.n_samples

  def get_random_sample(self):
    cur_idx = randint(0, len(self.X))
    return self.X[cur_idx], self.Y[cur_idx]

  def print_shapes(self):
    print('self.X : ', self.X.shape)
    print('self.Y : ', self.Y.shape)

from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=0)

train_dataset = EmbeddingsDataset(df_train, number_clusters)
train_dataset.print_shapes()

val_dataset = EmbeddingsDataset(df_val, number_clusters)
val_dataset.print_shapes()

test_dataset = EmbeddingsDataset(df_test, number_clusters)
test_dataset.print_shapes()

input_tensor = torch.randn(8, 4, 32)

## loading train dataset ##
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=512,
                          shuffle=True,
                          num_workers=2)

##loading test dataset ##
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=512,
                          shuffle=False,
                          num_workers=2)

##loading test dataset ##
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=512,
                          shuffle=False,
                          num_workers=2)

from torch.nn.modules.activation import Softmax
import torch

class FusionModelAdd(nn.Module):
  def __init__(self, num_classes=10, n_size=64):
    super(FusionModelAdd, self).__init__()

    self.linear_basic_3072 = nn.Sequential(
        nn.Linear(3072, n_size),
        nn.ReLU()
    )
    self.linear_basic_768 = nn.Sequential(
        nn.Linear(768, n_size),
        nn.ReLU()
    )

    self.linear_basic = nn.Sequential(
        nn.Linear(n_size, n_size),
        nn.ReLU()
    )

    self.classification = nn.Sequential(
      nn.Linear(n_size, num_classes),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    out_add1 = torch.add(x[:,0], x[:,1])
    out_add2 = torch.add(x[:,2], x[:,3])
    out = torch.add(out_add1, out_add2)
    out = self.linear_basic_768(out)
    out = self.linear_basic(out)
    out = self.classification(out)
    return out

from torch.nn.modules.activation import Softmax
import torch

class FusionModelConcat(nn.Module):
  def __init__(self, num_classes=10, n_size=64):
    super(FusionModelConcat, self).__init__()

    self.linear_basic_3072 = nn.Sequential(
        nn.Linear(3072, n_size),
        nn.ReLU()
    )

    self.linear_basic = nn.Sequential(
        nn.Linear(n_size, n_size),
        nn.ReLU()
    )

    self.classification = nn.Sequential(
      nn.Linear(n_size, num_classes),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    out = torch.cat((x[:,0], x[:,1], x[:,2], x[:,3]), dim=1)
    out = self.linear_basic_3072(out)
    out = self.linear_basic(out)
    out = self.classification(out)
    return out

from torch.nn.modules.activation import Softmax
import torch

class FusionModelMult(nn.Module):
  def __init__(self, num_classes=10, n_size=64):
    super(FusionModelMult, self).__init__()

    self.linear_basic_768 = nn.Sequential(
        nn.Linear(768, n_size),
        nn.ReLU()
    )

    self.linear_basic_768_2 = nn.Sequential(
        nn.Linear(n_size, n_size),
        nn.ReLU()
    )

    self.classification = nn.Sequential(
      nn.Linear(n_size, num_classes),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    out_mul1 = torch.mul(x[:,0], x[:,1])
    out_mul2 = torch.mul(x[:,2], x[:,3])
    out = torch.mul(out_mul1, out_mul2)
    out = self.linear_basic_768(out)
    out = self.linear_basic_768_2(out)
    out = self.classification(out)
    return out

from torch.nn.modules.activation import Softmax
import torch

class FusionModelAttSimple(nn.Module):
  def __init__(self, num_classes=10, n_size=64):
    super(FusionModelAttSimple, self).__init__()

    self.linear_basic_3072 = nn.Sequential(
        nn.Linear(3072, n_size),
        nn.ReLU()
    )
    self.linear_basic = nn.Sequential(
        nn.Linear(n_size, n_size),
        nn.ReLU()
    )
    self.layers_att_simple_1 = (
        nn.Linear(768, 1)
    )
    self.layers_att_simple_2 = (
        nn.Linear(768, 1)
    )

    self.layers_att_simple_3 = (
        nn.Linear(768, 1)
    )

    self.layers_att_simple_4 = (
        nn.Linear(768, 1)
    )

    self.apply_softmax = nn.Softmax(dim=1)
    self.classification = nn.Sequential(
      nn.Linear(n_size, num_classes),
      nn.Softmax(dim=1)
    )

  def forward(self, x):

    att_1 = self.layers_att_simple_1(x[:,0])
    att_2 = self.layers_att_simple_2(x[:,1])
    att_3 = self.layers_att_simple_3(x[:,2])
    att_4 = self.layers_att_simple_4(x[:,3])


    att_weight_simple = torch.stack([att_1, att_2, att_3, att_4])

    att_weights = self.apply_softmax(att_weight_simple)
    att_weights = att_weights.permute(1,0,2)

    x = x * att_weights

    out = torch.cat((x[:,0], x[:,1], x[:,2], x[:,3]), dim=1)
    out = self.linear_basic_3072(out)
    out = self.linear_basic(out)

    out = self.classification(out)
    return out

from torch.nn.modules.activation import Softmax
import torch

class FusionModelAttMultihead(nn.Module):
  def __init__(self, num_classes=10, n_size=64):
    super(FusionModelAttMultihead, self).__init__()

    self.linear_basic_3072 = nn.Sequential(
        nn.Linear(3072, n_size),
        nn.ReLU()
    )

    self.linear_basic = nn.Sequential(
        nn.Linear(n_size, n_size),
        nn.ReLU()
    )
    self.attn_layer = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)

    self.attn_layer_2 = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)

    self.classification = nn.Sequential(
      nn.Linear(n_size, num_classes),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    attn_output, attn_weights = self.attn_layer(x, x, x)
    attn_output, attn_weights = self.attn_layer(attn_output, attn_output, attn_output)

    out = torch.cat((attn_output[:,0], attn_output[:,1], attn_output[:,2], attn_output[:,3]), dim=1)
    out = self.linear_basic_3072(out)
    out = self.linear_basic(out)
    out = self.classification(out)
    return out

models = {
    "add" : FusionModelAdd(number_labels, number_of_layers),
    "cat" : FusionModelConcat(number_labels, number_of_layers),
    "mul" : FusionModelMult(number_labels, number_of_layers),
    "att_sim" : FusionModelAttSimple(number_labels, number_of_layers),
    "att_head" : FusionModelAttMultihead(number_labels, number_of_layers)
}

"""## Training loop

---
"""

def get_model_predictions(cur_model, loader_test):
  all_predictions = []
  cur_model.eval()
  with torch.no_grad():
    for i, (feat, lab) in tqdm(enumerate(loader_test), total=len(loader_test)):

      ## to device
      feat = feat.cuda()
      lab = lab.cuda()

      ## foward
      outputs = cur_model(feat)


      # torch.max retorna value, index
      _, predictions = torch.max(outputs.data, 1)
      all_predictions += predictions.cpu().numpy().tolist()
  return all_predictions

def get_metrics_f1_acc(y_pred, y_true, model_name,show=False):

  matrix_confusion = confusion_matrix(y_true, y_pred)

  dict_aux = {
      'accuracy': [metrics.accuracy_score(y_true, y_pred)],
      'f1-macro': [metrics.f1_score(y_true, y_pred, average='macro')],
      'f1-micro': [metrics.f1_score(y_true, y_pred, average='micro')],
      'recall macro'    : [metrics.recall_score(y_true, y_pred, average='macro')],
      'recall micro'    : [metrics.recall_score(y_true, y_pred, average='micro')],
      'precision macro' : [metrics.precision_score(y_true, y_pred, average='macro')],
      'precision micro' : [metrics.precision_score(y_true, y_pred, average='micro')],
      'confusion matrix': [matrix_confusion],
      'model_name' : [model_name]
  }
  if show:
    print('Confusion Matrix')
    print(matrix_confusion)

    sn.heatmap(matrix_confusion/matrix_confusion.sum(axis=1)[:,None], annot=True)
    return pd.DataFrame.from_dict(dict_aux)
  return metrics.f1_score(y_true, y_pred, average='macro'), pd.DataFrame.from_dict(dict_aux)

def get_prediction_and_save_model(cur_model, test_loader, best_f1, model_name):
  test_predictions = get_model_predictions(cur_model, test_loader)
  df_test[f"{model_name}_predictions"] = test_predictions
  cur_f1, cur_metrics = get_metrics_f1_acc(test_predictions, df_test.label, model_name)
  return cur_f1, cur_metrics

def fit_model(cur_model, num_epoch, loader_train, loader_val,  lr_dis=0.001, verbose=True, model_name=""):
  loss_dis = nn.CrossEntropyLoss()
  optimizer_dis = optim.Adam(cur_model.parameters(), lr=lr_dis)
  best_f1 = -1
  best_metrics = {}
  for epoch in range(num_epoch):
    cur_model.train()
    train_loss = 0
    for i, (feat, lab) in tqdm(enumerate(loader_train), total=len(loader_train)):

      optimizer_dis.zero_grad()

      ## to device
      feat = feat.cuda()
      lab = lab.cuda()

      ## foward
      outputs = cur_model(feat)

      loss = loss_dis(outputs, lab)

      ## backpropagation
      loss.backward()
      optimizer_dis.step()

      train_loss += loss.item()

    n_correct = 0
    n_samples = 0
    test_loss = 0

    cur_model.eval()
    with torch.no_grad():
      for i, (feat, lab) in tqdm(enumerate(loader_val), total=len(loader_val)):

        ## to device
        feat = feat.cuda()
        lab = lab.cuda()

        ## foward
        outputs = cur_model(feat)

        loss = loss_dis(outputs, lab)

        # torch.max retorna value, index
        _, predictions = torch.max(outputs.data, 1)
        n_samples += lab.shape[0]
        n_correct += (predictions == lab).sum().item()

        test_loss += loss.item()


      if verbose:
        acc = 100.0 * n_correct / n_samples
        print(f"\n Epoch {epoch+1}, loss_train: {train_loss/len(loader_train):.2f}, loss_test: {test_loss/len(loader_val):.2f}, acc_test: {acc:.2f}")
    cur_f1, cur_metrics =  get_prediction_and_save_model(cur_model, test_loader, best_f1, model_name)
    if best_f1 < cur_f1:
      best_f1 = cur_f1
      best_metrics = cur_metrics
      torch.save(cur_model, f"{model_path}/{model_name}.pt")
  return best_f1, best_metrics

all_best_metrics = []
for cur_model in models:
  print(f"stating training : {cur_model}")
  models[cur_model].cuda()
  _, best_model_metrics = fit_model(models[cur_model], number_of_training_epochs, train_loader, val_loader, lr_dis=learning_rate, model_name=cur_model)
  all_best_metrics.append(best_model_metrics)
  print("\n")
  print("#" * 10)

pd.concat(all_best_metrics)
pd.concat(all_best_metrics).to_pickle(results_path)
df_test.to_pickle(save_predictions)