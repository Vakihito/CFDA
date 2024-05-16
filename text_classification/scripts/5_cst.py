print("###############################")
print("##### Starting notebook 5 #####")
print("###############################")

import os

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
base_pred_path = os.environ['s0_save_predictions_all_data']

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

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

torch.manual_seed(random_state)
torch.cuda.manual_seed(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_state)
random.seed(random_state)

list_all_cluster_preds = []
for i in range(number_clusters):
  if (os.path.exists(f'{data_path}/model_cluster_{i}_prediction.pkl')):
    list_all_cluster_preds.append(pd.read_pickle(f'{data_path}/model_cluster_{i}_prediction.pkl'))

df_cluster_preds =  pd.concat(list_all_cluster_preds)
df_base_preds =  pd.read_pickle(base_pred_path)
df_cluster_preds = df_cluster_preds.sort_values(by='idx').reset_index(drop=True)
df_base_preds = df_base_preds.sort_values(by='idx').reset_index(drop=True)
df_cluster_preds.drop(columns=['label'],inplace=True)
df_base_preds.drop(columns=['label'], inplace=True)

df_add_train = df_base_preds[df_base_preds['prediction'] == df_cluster_preds['prediction']].reset_index(drop=True)
df_add_train['label'] = list(df_add_train['prediction'].values) # adding the pseudo label

idx_to_add_to_train = df_add_train['idx'].values

df_test = pd.read_pickle(test_path)
df_train = pd.read_pickle(train_path)
df_train, df_val = train_test_split(df_train, test_size=0.5, random_state=0,stratify=df_train['dataset'])

df_test_add_train = df_test[df_test.idx.isin(idx_to_add_to_train)].reset_index(drop=True)
df_add_train.sort_values(by='idx')
df_test_add_train.sort_values(by='idx')
df_test_add_train['label'] = list(df_add_train['prediction'].values) # setting the pseudo label

df_train = pd.concat([df_train, df_test_add_train])

df_target = df_test[~df_test['idx'].isin(df_train['idx'])].reset_index(drop=True)

import numpy as np
df_train['cluster_labels'] = df_train['cluster_labels'].apply(np.int64)
df_val['cluster_labels'] = df_val['cluster_labels'].apply(np.int64)
df_test['cluster_labels'] = df_test['cluster_labels'].apply(np.int64)
df_target['cluster_labels'] = df_target['cluster_labels'].apply(np.int64)

number_labels = len(df_train['label'].unique())
n_clusters = len(df_train['cluster_labels'].unique())

class EmbeddingsDataset(Dataset):
  def __init__(self, cur_df, number_clusters):
    self.X = np.array(cur_df[[f'embeddings_cluster_{i}' for i in range(number_clusters)]].values.tolist())
    self.Y = torch.from_numpy(cur_df["label"].values)
    self.clusters = torch.from_numpy(cur_df["cluster_labels"].values)

    ## define numero de samples de cada split
    self.n_samples = len(self.Y)

  def __getitem__(self, index):
    return self.X[index], self.Y[index], self.clusters[index]

  def __len__(self):
    return self.n_samples

  def get_random_sample(self):
    cur_idx = randint(0, len(self.X))
    return self.X[cur_idx], self.Y[cur_idx], self.clusters[cur_idx]

  def print_shapes(self):
    print('self.X : ', self.X.shape)
    print('self.Y : ', self.Y.shape)

class ReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class FusionModelConcat(nn.Module):
  def __init__(self, num_classes=10, n_size=64):
    super(FusionModelConcat, self).__init__()

    self.linear_basic_3072 = nn.Sequential(
        nn.Linear(3072, n_size),
        nn.BatchNorm1d(n_size),
        nn.ReLU()
    )

    self.linear_basic = nn.Sequential(
        nn.Linear(n_size, n_size),
        nn.BatchNorm1d(n_size),
        nn.ReLU()
    )

    ## sentiment classification linear ##
    self.classification = nn.Sequential(
      nn.Linear(n_size, num_classes),
      nn.Softmax(dim=1)
    )

    ## domain classification linear ##
    self.domain_class = nn.Sequential(
      nn.Linear(n_size, n_clusters),
    )

    ## Embedding reconstruction ###
    self.reconstruction_layer_1 = nn.Sequential(
      nn.Linear(n_size, n_size // 4),
      nn.BatchNorm1d(n_size // 4),
      nn.PReLU()
    )

    self.reconstruction_layer_2 = nn.Sequential(
      nn.Linear(n_size // 4, n_size),
      nn.BatchNorm1d(n_size),
      nn.PReLU()
    )

  def forward(self, x, alpha):
    out = torch.cat((x[:,0], x[:,1], x[:,2], x[:,3]), dim=1)
    out = self.linear_basic_3072(out)
    embedding_output = self.linear_basic(out)

    # foward sentiment
    classification_output = self.classification(embedding_output)

    # foward domain
    reverse_embbedding_layer = ReverseLayer.apply(embedding_output, alpha)
    domain_output = self.domain_class(reverse_embbedding_layer)

    # foward reconstruction
    x_reconstruction = self.reconstruction_layer_1(embedding_output)
    reconstruction_output = self.reconstruction_layer_2(x_reconstruction)

    return (classification_output, domain_output, embedding_output, reconstruction_output)

class FusionModelAttMultihead(nn.Module):
  def __init__(self, num_classes=10, n_size=64):
    super(FusionModelAttMultihead, self).__init__()

    self.linear_basic_3072 = nn.Sequential(
        nn.Linear(3072, n_size),
        nn.BatchNorm1d(n_size),
        nn.PReLU()
    )

    self.linear_basic = nn.Sequential(
        nn.Linear(n_size, n_size),
        nn.BatchNorm1d(n_size),
        nn.PReLU()
    )
    self.attn_layer = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)

    ## sentiment classification linear ##
    self.classification = nn.Sequential(
      nn.Linear(n_size, num_classes),
      nn.Softmax(dim=1)
    )

    ## domain classification linear ##
    self.domain_class = nn.Sequential(
      nn.Linear(n_size, n_clusters),
    )

    ## Embedding reconstruction ###
    self.reconstruction_layer_1 = nn.Sequential(
      nn.Linear(n_size, n_size // 4),
      nn.BatchNorm1d(n_size // 4),
      nn.PReLU()
    )

    self.reconstruction_layer_2 = nn.Sequential(
      nn.Linear(n_size // 4, n_size),
      nn.BatchNorm1d(n_size),
      nn.PReLU()
    )

  def forward(self, x, alpha):
    attn_output, attn_weights = self.attn_layer(x, x, x)

    out = torch.cat((attn_output[:,0], attn_output[:,1], attn_output[:,2], attn_output[:,3]), dim=1)
    out = self.linear_basic_3072(out)
    embedding_output = self.linear_basic(out)

    # foward sentiment class
    classification_output = self.classification(embedding_output)

    # foward domain class
    reverse_embbedding_layer = ReverseLayer.apply(embedding_output, alpha)
    domain_output = self.domain_class(reverse_embbedding_layer)

    # foward reconstruction
    x_reconstruction = self.reconstruction_layer_1(embedding_output)
    reconstruction_output = self.reconstruction_layer_2(x_reconstruction)

    return (classification_output, domain_output, embedding_output, reconstruction_output)

train_dataset = EmbeddingsDataset(df_train, number_clusters)
val_dataset = EmbeddingsDataset(df_val, number_clusters)
test_dataset = EmbeddingsDataset(df_test, number_clusters)
target_dataset = EmbeddingsDataset(df_target, number_clusters)

g = torch.Generator()
g.manual_seed(0)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=4096,
                          shuffle=True,
                          num_workers=2,
                          generator=g)

##loading test dataset ##
target_loader = DataLoader(dataset=target_dataset,
                          batch_size=4096,
                          shuffle=True,
                          num_workers=2,
                          generator=g)

##loading test dataset ##
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=4096,
                          shuffle=False,
                          num_workers=2)

##loading test dataset ##
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=4096,
                          shuffle=False,
                          num_workers=2)

models = {
    "cat" : FusionModelConcat(number_labels, number_of_layers),
    "att_head" : FusionModelAttMultihead(number_labels, number_of_layers)
}

def get_model_predictions(cur_model, loader_test):
  all_predictions = []
  cur_model.eval()
  with torch.no_grad():
    for i, (feat, lab, _) in enumerate(loader_test):

      ## to device
      feat = feat.to(device)
      lab = lab.to(device)

      ## foward
      outputs, _, _, _ = cur_model(feat, 0)

      # torch.max retorna value, index
      _, predictions = torch.max(outputs.data, 1)
      all_predictions += predictions.cpu().numpy().tolist()
  return all_predictions

def get_metrics_f1_acc(y_pred, y_true, model_name):

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

  return metrics.f1_score(y_true, y_pred, average='macro'), pd.DataFrame.from_dict(dict_aux)

def get_prediction_and_save_model(cur_model, test_loader, model_name):
  test_predictions = get_model_predictions(cur_model, test_loader)
  df_test[f"{model_name}_predictions"] = test_predictions
  cur_f1, cur_metrics = get_metrics_f1_acc(test_predictions, df_test.label, model_name)
  return cur_f1, cur_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

THETA = 0.01
GAMMA = 10 # EFFECT OF THE DOMAIN ADAPTATION LOSS 10 IS THE DEFAULT

BETHA = 1

try:
  os.mkdir(model_path)
except:
  print("model dir already exists")
  pass

criterion_sentiment = nn.CrossEntropyLoss()
criterion_adversarial = nn.CrossEntropyLoss()
criterion_reconstruction = nn.MSELoss()
def fit_model(cur_model, num_epoch, loader_train, loader_test, loader_target,  lr_dis=0.001, verbose=True, model_name=""):


  optimizer_dis = optim.Adam(cur_model.parameters(), lr=lr_dis)
  best_f1 = -1
  best_metrics = {}
  len_dataloader = len(loader_train) + len(loader_target)
  with tqdm(total=(num_epoch) * (len_dataloader)) as pbar:
    for epoch in range(num_epoch):
      cur_model.train()
      train_loss = 0

      ##########################
      ### FORWARD Source DATA ##
      ##########################
      for i, (feat, lab, clusters) in enumerate(loader_train):
        optimizer_dis.zero_grad()
        p = (i + epoch * len(loader_train)) / (num_epoch * len(loader_train))
        alpha  = 2. / (1. + np.exp(-GAMMA * p)) - 1

        ## to device
        feat = feat.to(device)
        lab = lab.to(device)
        clusters = clusters.to(device)

        outputs_sentiment, outputs_domain_source, output_emb, output_emb_rec = cur_model(feat, alpha)

        loss_sent = criterion_sentiment(outputs_sentiment, lab)
        loss_domain_source = criterion_adversarial(outputs_domain_source, clusters)
        loss_rec_source = criterion_reconstruction(output_emb, output_emb_rec)

        loss = loss_sent + (loss_domain_source) * THETA + (loss_rec_source) * BETHA
        ## backpropagation
        loss.backward()
        optimizer_dis.step()
        train_loss += loss.item()
        pbar.update(1)

      ##########################
      ### FORWARD Target DATA ##
      ##########################
      for i, (feat, _, clusters) in enumerate(loader_target):
        optimizer_dis.zero_grad()
        p = (i + epoch * len(loader_target)) / (num_epoch * len(loader_target))
        alpha  = 2. / (1. + np.exp(-GAMMA * p)) - 1

        ## to device
        feat = feat.to(device)
        clusters = clusters.to(device)

        _, outputs_domain_target, output_emb_target, output_emb_rec_target = cur_model(feat, alpha)

        loss_domain_target = criterion_adversarial(outputs_domain_target, clusters)
        loss_rec_target = criterion_reconstruction(output_emb_target, output_emb_rec_target)


        loss = (loss_domain_target) * THETA + (loss_rec_target) * BETHA
        ## backpropagation
        loss.backward()
        optimizer_dis.step()
        train_loss += loss.item()
        pbar.update(1)

      print(f"\n Epoch {epoch+1}, loss_train: {train_loss/(len(loader_train) + len(loader_test)):.2f}")
      cur_f1, cur_metrics =  get_prediction_and_save_model(cur_model, test_loader, model_name)
      if best_f1 < cur_f1:
        best_f1 = cur_f1
        best_metrics = cur_metrics
        print("saving best metrics")
        torch.save(cur_model, f"{model_path}/{model_name}_fusion_triple_loss.pt")
  return best_f1, best_metrics

all_best_metrics = []
for cur_model in models:
  print(f"stating training : {cur_model}")
  models[cur_model] = models[cur_model].to(device)
  _, best_model_metrics = fit_model(models[cur_model], number_of_training_epochs, train_loader, test_loader, target_loader, lr_dis=learning_rate, model_name=cur_model)
  all_best_metrics.append(best_model_metrics)
  print("\n")
  print("#" * 10)

df_fusion_metrics = pd.concat(all_best_metrics)
df_fusion_metrics

df_fusion_metrics.to_pickle(results_path)
df_test.to_pickle(save_predictions)