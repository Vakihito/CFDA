print("###############################")
print("##### Starting notebook 5 #####")
print("###############################")
import os
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

batch_size = int(1024 * 4)
num_epochs = int(os.environ['fusion_training_epochs'])
learning_rate = float(os.environ['fusion_training_lr'])

save_path = os.environ['save_path']
images_path = os.environ["read_data_path"]
test_dataset = os.environ["dataset_test"]
n_clusters = int(os.environ["n_clusters"]) + 1
output_model_path = os.environ['output_model_path']
fusion_pacience = int(os.environ['fusion_pacience'])

os.mkdir(output_model_path)

df_test = pd.read_pickle(f"{save_path}/test_data_clustered.pkl")
df_test = df_test[["dataset", "index","image_name",	"data_path",	"embeddings",	"cluster_labels"]].reset_index(drop=True)
df_simple = pd.read_pickle(f'{save_path}/simple_predictions.pkl')

# loading the cluster predictions #
list_dfs_cluster = []
for cur_cluster in range(n_clusters-1):
  cur_df = pd.read_pickle(f'{save_path}/cluster_{cur_cluster}_predictions.pkl')
  list_dfs_cluster.append(cur_df)
df_cluster = pd.concat(list_dfs_cluster)
###

df_simple.sort_values(by='index', inplace=True)
df_cluster.sort_values(by='index', inplace=True)
df_test.sort_values(by='index', inplace=True)

index_aggree = (df_simple['prediction'].values == df_cluster['prediction'].values)
df_agree = df_simple[index_aggree].reset_index(drop=True)
map_pred_idx = {row['index']: row['prediction'] for _, row in df_agree.iterrows()}
df_test_agree = df_test[index_aggree].reset_index(drop=True)

df_test_agree['sentiment'] = df_test_agree['index'].apply(lambda x: map_pred_idx[x])

df_train = pd.read_pickle(f"{save_path}/train_data_clustered.pkl")
df_val = pd.read_pickle(f"{save_path}/val_data_clustered.pkl")
df_test = pd.read_pickle(f"{save_path}/test_data_clustered.pkl")

df_train = pd.concat([df_test_agree, df_train]).reset_index(drop=True)

def add_embedding_to_map(cur_df, cur_cluster, cur_map):
  for _, row in cur_df.iterrows():
    cur_map[row['index']] = row[f"cluster_{cur_cluster}" ]
  return cur_map

def create_map_emb(cur_cluster, map_index_emb):
  cur_df_train = pd.read_pickle(f"{save_path}/train_cluster_{cur_cluster}_emb.pkl")
  cur_df_val = pd.read_pickle(f"{save_path}/val_cluster_{cur_cluster}_emb.pkl")
  cur_df_test = pd.read_pickle(f"{save_path}/test_cluster_{cur_cluster}_emb.pkl")

  add_embedding_to_map(cur_df_train, cur_cluster, map_index_emb)
  add_embedding_to_map(cur_df_val, cur_cluster, map_index_emb)
  add_embedding_to_map(cur_df_test, cur_cluster, map_index_emb)

  df_train[f"cluster_{cur_cluster}"] = df_train['index'].apply(lambda x: map_index_emb[x])
  df_val[f"cluster_{cur_cluster}"] = df_val['index'].apply(lambda x: map_index_emb[x])
  df_test[f"cluster_{cur_cluster}"] = df_test['index'].apply(lambda x: map_index_emb[x])

for cur_cluster in range(n_clusters):
  map_index_emb = {}
  create_map_emb(cur_cluster, map_index_emb)

  df_train[f"cluster_{cur_cluster}"] = df_train['index'].apply(lambda x: map_index_emb[x])
  df_val[f"cluster_{cur_cluster}"] = df_val['index'].apply(lambda x: map_index_emb[x])
  df_test[f"cluster_{cur_cluster}"] = df_test['index'].apply(lambda x: map_index_emb[x])

import torch
import torch.nn as nn

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.list_embeddings = np.array([[cur_row[f'cluster_{cur_cluster}'].tolist() for cur_cluster in range(n_clusters)] for idx, cur_row in dataframe.iterrows()], dtype=np.float32)
        self.list_cluster = np.array(dataframe['cluster_labels'].values, dtype=np.float32)
        self.list_sentiment = dataframe['sentiment'].values
        self.list_index = dataframe['index'].values
        self.list_domain = np.array(dataframe['cluster_labels'].values, dtype=np.int64)

    def __len__(self):
        return len(self.list_sentiment)

    def __getitem__(self, idx):
        cur_list_embeddings = self.list_embeddings[idx]
        cur_cluster_label = self.list_cluster[idx]
        cur_sentiment_label = self.list_sentiment[idx]
        cur_index = self.list_index[idx]
        cur_domain = self.list_domain[idx]

        sample = {
            "emb" : cur_list_embeddings,
            "cluster": cur_cluster_label,
            "sentiment": cur_sentiment_label,
            "index" : cur_index,
            "domain" : cur_domain
        }

        return sample

def custom_collate(batch):
    data_batch = [item['emb'] for item in batch]
    cluster_batch = [item['cluster'] for item in batch]
    sentiment_batch = [item['sentiment'] for item in batch]
    index_batch = [item['index'] for item in batch]
    domain_batch = [item['domain'] for item in batch]


    # Stack the data NumPy arrays along a new axis to form a single N-dimensional array
    stacked_data = np.stack(data_batch, axis=0)
    return torch.from_numpy(stacked_data), torch.tensor(cluster_batch), torch.tensor(sentiment_batch), index_batch, torch.tensor(domain_batch)

df_train.head(3)

df_target = df_test[~df_test['index'].isin(df_train['index'].unique())].reset_index(drop=True)

g = torch.Generator()
g.manual_seed(0)

train_dataset = CustomDataset(dataframe=df_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=custom_collate,
                              num_workers=2, generator=g)

target_dataset = CustomDataset(dataframe=df_target)
target_dataloader = DataLoader(target_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=custom_collate,
                              num_workers=2, generator=g)


test_dataset = CustomDataset(dataframe=df_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=custom_collate,num_workers=2)

val_dataset = CustomDataset(dataframe=df_val)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=custom_collate, num_workers=2)

class ReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class ClassificationModel(nn.Module):
    def __init__(self, d_model, inter_layer_size=64, n_classes=3, n_domains=6, alpha=0.1):
        super(ClassificationModel, self).__init__()


        self.att_layer = nn.MultiheadAttention(d_model, num_heads=64, batch_first=True)
        self.single_dense = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.PReLU()
        )
        self.linear_basic_betw = nn.Sequential(
          nn.Linear(d_model * n_clusters, inter_layer_size),
          nn.BatchNorm1d(inter_layer_size),
          nn.PReLU()
        )

        ### Sentiment Classification model layers ###
        self.linear_basic_betw_2 = nn.Sequential(
          nn.Linear(inter_layer_size, inter_layer_size),
          nn.BatchNorm1d(inter_layer_size),
          nn.PReLU()
        )

        self.linear_basic_betw_3 = nn.Sequential(
          nn.Linear(inter_layer_size, inter_layer_size),
          nn.BatchNorm1d(inter_layer_size),
          nn.Dropout(p=0.5),
          nn.PReLU()
        )

        self.fc_class = nn.Sequential(
            nn.Linear(inter_layer_size, n_classes),
        )
        #############################################
        ######### Embedding reconstruction ##########
        #############################################
        self.reconstruction_layer_1 = nn.Sequential(
          nn.Linear(inter_layer_size, inter_layer_size // 4),
          nn.PReLU()
        )

        self.reconstruction_layer_2 = nn.Sequential(
          nn.Linear(inter_layer_size // 4, inter_layer_size),
          nn.PReLU()
        )

        ## domain classification linear ##
        self.domain_class = nn.Sequential(
          nn.Linear(inter_layer_size, n_domains),
        )

        self.dropout = nn.Dropout(p=0.5)


    def forward(self, inputs_list=[], alpha=0):
        output_list = []
        attn_output, _ = self.att_layer(inputs_list, inputs_list, inputs_list)
        for i in range(n_clusters):
          tmp_attn = self.single_dense(attn_output[:,i])
          output_list.append(tmp_attn * inputs_list[:,i] + inputs_list[:,i])

        concat_emb = torch.cat(output_list, dim=-1)
        embbedding_layer_1 = self.linear_basic_betw(concat_emb)
        embbedding_layer_2 = self.linear_basic_betw_2(embbedding_layer_1)
        embbedding_layer_3 = self.linear_basic_betw_3(embbedding_layer_1 + embbedding_layer_2)
        ### foward sentiment classification ###
        x_sent = self.linear_basic_betw_2(embbedding_layer_3)
        sent_output = self.fc_class(embbedding_layer_3)
        ###

        ### Embedding reconstruction ###
        x_reconstruction = self.reconstruction_layer_1(embbedding_layer_3)
        reconstruction_output = self.reconstruction_layer_2(x_reconstruction)


        ### foward domain classification ##
        reverse_embbedding_layer = ReverseLayer.apply(embbedding_layer_3, alpha)
        domain_output = self.domain_class(reverse_embbedding_layer)

        return sent_output, reconstruction_output, embbedding_layer_3, domain_output

class ClassificationModelConcat(nn.Module):
    def __init__(self, d_model, inter_layer_size=64, n_classes=3, n_domains=6, alpha=0.1):
        super(ClassificationModelConcat, self).__init__()


        self.att_layer = nn.MultiheadAttention(d_model, num_heads=64, batch_first=True)
        self.single_dense = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.PReLU()
        )
        self.linear_basic_betw = nn.Sequential(
          nn.Linear(d_model * n_clusters, inter_layer_size),
          nn.BatchNorm1d(inter_layer_size),
          nn.PReLU()
        )

        ### Sentiment Classification model layers ###
        self.linear_basic_betw_2 = nn.Sequential(
          nn.Linear(inter_layer_size, inter_layer_size),
          nn.BatchNorm1d(inter_layer_size),
          nn.Dropout(p=0.5),
          nn.PReLU()
        )

        self.linear_basic_betw_3 = nn.Sequential(
          nn.Linear(inter_layer_size, inter_layer_size),
          nn.BatchNorm1d(inter_layer_size),
          nn.PReLU()
        )

        self.linear_basic_betw_4 = nn.Sequential(
          nn.Linear(inter_layer_size, inter_layer_size),
          nn.BatchNorm1d(inter_layer_size),
          nn.Dropout(p=0.5),
          nn.PReLU()
        )


        self.linear_sent_1 = nn.Sequential(
          nn.Linear(inter_layer_size, inter_layer_size),
          nn.BatchNorm1d(inter_layer_size),
          nn.PReLU()
        )

        self.linear_sent_2 = nn.Sequential(
          nn.Linear(inter_layer_size, inter_layer_size),
          nn.BatchNorm1d(inter_layer_size),
          nn.Dropout(p=0.5),
          nn.PReLU()
        )


        self.fc_class = nn.Sequential(
            nn.Linear(inter_layer_size, n_classes),
        )
        #############################################
        ######### Embedding reconstruction ##########
        #############################################
        self.reconstruction_layer_1 = nn.Sequential(
          nn.Linear(inter_layer_size, inter_layer_size // 4),
          nn.PReLU()
        )

        self.reconstruction_layer_2 = nn.Sequential(
          nn.Linear(inter_layer_size // 4, inter_layer_size),
          nn.PReLU()
        )

        ## domain classification linear ##
        self.domain_class = nn.Sequential(
          nn.Linear(inter_layer_size, n_domains),
        )


    def forward(self, inputs_list=[], alpha=0):
        output_list = []
        concat_emb = torch.flatten(inputs_list, start_dim=1)
        embbedding_layer_1 = self.linear_basic_betw(concat_emb)
        embbedding_layer_2 = self.linear_basic_betw_2(embbedding_layer_1)
        embbedding_layer_3 = self.linear_basic_betw_3(embbedding_layer_1 + embbedding_layer_2)
        embbedding_layer_4 = self.linear_basic_betw_4(embbedding_layer_3 + embbedding_layer_2)


        ### foward sentiment classification ###
        sent_emb_1 = self.linear_sent_1(embbedding_layer_4)
        sent_emb_2 = self.linear_sent_2(sent_emb_1)
        sent_output = self.fc_class(sent_emb_1 + sent_emb_2)
        ###

        ### Embedding reconstruction ###
        x_reconstruction = self.reconstruction_layer_1(embbedding_layer_4)
        reconstruction_output = self.reconstruction_layer_2(x_reconstruction)


        ### foward domain classification ##
        reverse_embbedding_layer = ReverseLayer.apply(embbedding_layer_4, alpha)
        domain_output = self.domain_class(reverse_embbedding_layer)

        return sent_output, reconstruction_output, embbedding_layer_4, domain_output

class ClassificationModelSimpleMultihead(nn.Module):
    def __init__(self, d_model, inter_layer_size=64, n_classes=3, n_domains=6, alpha=0.1):
        super(ClassificationModelSimpleMultihead, self).__init__()


        self.att_layer = nn.MultiheadAttention(d_model, num_heads=64, batch_first=True)
        self.single_dense = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.PReLU()
        )
        self.linear_basic_betw = nn.Sequential(
          nn.Linear(d_model * n_clusters, inter_layer_size),
          nn.BatchNorm1d(inter_layer_size),
          nn.PReLU()
        )

        ### Sentiment Classification model layers ###
        self.linear_basic_betw_2 = nn.Sequential(
          nn.Linear(inter_layer_size, inter_layer_size),
          nn.BatchNorm1d(inter_layer_size),
          nn.PReLU()
        )

        self.linear_basic_betw_3 = nn.Sequential(
          nn.Linear(inter_layer_size, inter_layer_size),
          nn.BatchNorm1d(inter_layer_size),
          nn.PReLU()
        )

        self.fc_class = nn.Sequential(
            nn.Linear(inter_layer_size, n_classes),
        )
        #############################################
        ######### Embedding reconstruction ##########
        #############################################
        self.reconstruction_layer_1 = nn.Sequential(
          nn.Linear(inter_layer_size, inter_layer_size // 4),
          nn.PReLU()
        )

        self.reconstruction_layer_2 = nn.Sequential(
          nn.Linear(inter_layer_size // 4, inter_layer_size),
          nn.PReLU()
        )

        ## domain classification linear ##
        self.domain_class = nn.Sequential(
          nn.Linear(inter_layer_size, n_domains),
        )


    def forward(self, inputs_list=[], alpha=0):
        output_list = []
        attn_output, _ = self.att_layer(inputs_list, inputs_list, inputs_list)
        concat_emb = torch.flatten(attn_output, start_dim=1)
        embbedding_layer_1 = self.linear_basic_betw(concat_emb)
        embbedding_layer_2 = self.linear_basic_betw_2(embbedding_layer_1)
        embbedding_layer_3 = self.linear_basic_betw_3(embbedding_layer_1 + embbedding_layer_2)
        ### foward sentiment classification ###
        x_sent = self.linear_basic_betw_2(embbedding_layer_3)
        sent_output = self.fc_class(embbedding_layer_3)
        ###

        ### Embedding reconstruction ###
        x_reconstruction = self.reconstruction_layer_1(embbedding_layer_3)
        reconstruction_output = self.reconstruction_layer_2(x_reconstruction)


        ### foward domain classification ##
        reverse_embbedding_layer = ReverseLayer.apply(embbedding_layer_3, alpha)
        domain_output = self.domain_class(reverse_embbedding_layer)

        return sent_output, reconstruction_output, embbedding_layer_3, domain_output

from sklearn.metrics import classification_report,confusion_matrix
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn import metrics
import seaborn as sns
import os

def validation_metrics(y_true, y_pred, show_img=True):
  dict_aux = {
      'accuracy' :[ metrics.accuracy_score(y_true, y_pred)],
      'f1-macro' : [metrics.f1_score(y_true, y_pred, average='macro')],
      'f1-micro' : [metrics.f1_score(y_true, y_pred, average='micro')],
      'recall macro'    : [metrics.recall_score(y_true, y_pred, average='macro')],
      'recall micro'    : [metrics.recall_score(y_true, y_pred, average='micro')],
      'precision macro' : [metrics.precision_score(y_true, y_pred, average='macro')],
      'precision micro' : [metrics.precision_score(y_true, y_pred, average='micro')]

  }
  return dict_aux

def evaluate_dataset(model, cur_data_loader, cur_split="test"):
  model.eval()
  correct = 0
  total = 0
  total_loss = 0.0
  list_labels = []
  list_predictions = []
  list_all_indices = []
  with torch.no_grad():
      for cur_list_embeddings, cur_cluster_label, cur_sentiment_label, cur_index, cur_domain in cur_data_loader:
          l_indexes = list(cur_index)

          cur_list_embeddings, cur_cluster_label = cur_list_embeddings.to(device), cur_cluster_label.to(device)
          cur_domain = cur_domain.to(device)
          cur_sentiment_label = cur_sentiment_label.to(device)

          sent_outputs, _, _, _ = model(cur_list_embeddings)

          sent_loss = criterion_sent(sent_outputs, cur_sentiment_label)
          loss = sent_loss


          _, predicted = torch.max(sent_outputs.data, 1)

          list_labels.append(cur_sentiment_label.detach().cpu().numpy())
          list_predictions.append(predicted.detach().cpu().numpy())

          list_all_indices += l_indexes
          total_loss += loss.item()
  val_loss = total_loss/len(cur_data_loader)
  y_true = np.concatenate(list_labels).tolist()
  y_pred = np.concatenate(list_predictions).tolist()
  prediction_metrics = validation_metrics(y_true, y_pred)
  val_f1 = prediction_metrics['f1-macro']
  print("\n")
  print(f"{cur_split} Metrics")
  for cur_metric in prediction_metrics:
    print(f"{cur_metric} : {prediction_metrics[cur_metric]}")
  return val_f1[0], y_pred, list_all_indices, val_loss, prediction_metrics



emb_dim = len(df_train.iloc[0]['cluster_0'])
print(f"embedding dim : {emb_dim}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


criterion_sent = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()
reconstruction_criterion = nn.MSELoss()

models_dict = {
    'cst_simple_multihead': ClassificationModelSimpleMultihead(emb_dim, inter_layer_size=512, n_classes=3, n_domains=2),
    'cst_att' : ClassificationModel(emb_dim, inter_layer_size=512, n_classes=3, n_domains=2),
    'cst_concat' : ClassificationModelConcat(emb_dim, inter_layer_size=512, n_classes=3, n_domains=2),
}

def train_model(model,model_name):
  THETA = 0.0001
  GAMMA = 10 # EFFECT OF THE DOMAIN ADAPTATION LOSS 10 IS THE DEFAULT
  BETHA = 0.001

  best_f1_score = 0
  best_predictions = []
  cur_training_pacience = 0
  best_val_loss = float('inf')
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

  for epoch in tqdm(range(num_epochs)):
      model.train()
      total_loss = 0.0
      for i, (cur_list_embeddings, cur_cluster_label, cur_sentiment_label, cur_index, cur_domain) in enumerate(train_dataloader):
          p = (i + epoch * len(train_dataloader)) / (num_epochs * len(train_dataloader))
          alpha  = 2. / (1. + np.exp(-GAMMA * p))

          l_indexes = list(cur_index)
          cur_list_embeddings, cur_cluster_label = cur_list_embeddings.to(device), cur_cluster_label.to(device)
          cur_sentiment_label = cur_sentiment_label.to(device)
          cur_domain = cur_domain.to(device)


          optimizer.zero_grad()
          sent_outputs, reconstruction_output, embbedding_layer, domain_output = model(cur_list_embeddings, alpha)


          sent_loss = criterion_sent(sent_outputs, cur_sentiment_label)
          reconstruction_loss = reconstruction_criterion(reconstruction_output, embbedding_layer)
          domain_loss = criterion_domain(domain_output, cur_domain)

          loss = sent_loss + (reconstruction_loss * BETHA) + (domain_loss * THETA)

          loss.backward()
          optimizer.step()
          total_loss += loss.item()


      ##########################
      ### FORWARD Target DATA ##
      ##########################
      for i, (cur_list_embeddings, cur_cluster_label, cur_sentiment_label, cur_index, cur_domain) in enumerate(target_dataloader):
          optimizer.zero_grad()
          p = (i + epoch * len(train_dataloader)) / (num_epochs * len(train_dataloader))
          alpha  = 2. / (1. + np.exp(-GAMMA * p))

          l_indexes = list(cur_index)
          cur_list_embeddings, cur_cluster_label = cur_list_embeddings.to(device), cur_cluster_label.to(device)

          cur_sentiment_label = cur_sentiment_label.to(device)
          cur_domain = cur_domain.to(device)

          _, reconstruction_output, embbedding_layer, domain_output = model(cur_list_embeddings, alpha)

          reconstruction_loss = reconstruction_criterion(reconstruction_output, embbedding_layer)
          domain_loss = criterion_domain(domain_output, cur_domain)


          loss = (reconstruction_loss * BETHA) + (domain_loss * THETA)

          ## backpropagation
          loss.backward()
          optimizer.step()
          total_loss += loss.item()

      print(f'\n Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader)}')

      val_f1_score, cur_y_pred, list_all_indices, cur_val_loss, cur_val_metrics = evaluate_dataset(model,val_dataloader)
      if val_f1_score > best_f1_score:
        best_f1_score = val_f1_score
        print(f"updating f1 best score {best_f1_score}")
        torch.save(model.state_dict(), f'{output_model_path}/{model_name}_cst.pt')
        best_predictions = cur_y_pred
        cur_training_pacience = 0


      if best_val_loss > cur_val_loss:
        best_val_loss = cur_val_loss
        print(f"Updating best validation loss : {cur_val_loss}")

      else:
        cur_training_pacience += 1
      print(f"Current val Loss : {cur_val_loss}")


      if cur_training_pacience > fusion_pacience:
        print("Early stopping triggered !!")
        break


      print("\n")
      print("=" * 30)
      print("\n")
  model.load_state_dict(torch.load(f'{output_model_path}/{model_name}_cst.pt'))
  val_f1_score, cur_y_pred, list_all_indices, _, cur_test_metrics = evaluate_dataset(model,test_dataloader)
  display(pd.DataFrame(cur_test_metrics))
  return pd.DataFrame(cur_test_metrics)

list_metrics =[]
for cur_model_name in models_dict:
  cur_model = models_dict[cur_model_name].to(device)
  test_metrics = train_model(cur_model, cur_model_name)
  test_metrics['type'] = cur_model_name
  list_metrics.append(test_metrics)

metrics_cst = pd.concat(list_metrics)
metrics_cst.to_pickle(f'{save_path}/cluster_fusion_predictions_cluster_att.pkl')