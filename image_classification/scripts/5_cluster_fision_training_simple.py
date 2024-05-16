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
domain_alpha = float(os.environ['training_alpha'])

df_train = pd.read_pickle(f"{save_path}/train_data_clustered_ensamble.pkl")
df_val = pd.read_pickle(f"{save_path}/val_data_clustered.pkl")
df_test = pd.read_pickle(f"{save_path}/test_data_clustered.pkl")

for cur_cluster in range(n_clusters):
  cur_df_train_cluster = pd.read_pickle(f"{save_path}/train_cluster_{cur_cluster}_emb.pkl")
  cur_df_val_cluster = pd.read_pickle(f"{save_path}/val_cluster_{cur_cluster}_emb.pkl")
  cur_df_test_cluster = pd.read_pickle(f"{save_path}/test_cluster_{cur_cluster}_emb.pkl")

  df_train[f"cluster_{cur_cluster}"] = cur_df_train_cluster[f"cluster_{cur_cluster}"]
  df_val[f"cluster_{cur_cluster}"] = cur_df_val_cluster[f"cluster_{cur_cluster}"]
  df_test[f"cluster_{cur_cluster}"] = cur_df_test_cluster[f"cluster_{cur_cluster}"]

# mapping all domains to a value
domain_mapping = {}
domain_counter = 0
list_all_domains = list(df_train["dataset"].unique()) + list(df_val["dataset"].unique()) + list(df_test["dataset"].unique())

for cur_domain in list_all_domains:
  if not cur_domain in domain_mapping:
    domain_mapping[cur_domain] = domain_counter
    domain_counter += 1

df_train['domain'] = df_train['dataset'].apply(lambda x: domain_mapping[x])
df_val['domain'] = df_val['dataset'].apply(lambda x: domain_mapping[x])
df_test['domain'] = df_test['dataset'].apply(lambda x: domain_mapping[x])

df_train.head(3)

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
        self.list_domain = dataframe["domain"].values

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

g = torch.Generator()
g.manual_seed(0)

train_dataset = CustomDataset(dataframe=df_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
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

        self.linear_basic_betw = nn.Sequential(
          nn.Linear(d_model * n_clusters, inter_layer_size),
          nn.PReLU()
        )

        self.fc_between = nn.Sequential(
            nn.Linear(inter_layer_size, inter_layer_size),
            nn.ReLU()
        )

        self.fc_class = nn.Sequential(
            nn.Linear(inter_layer_size, 3),
        )

        # for each layer cluster embedding return the attention weights
        self.cluster_attention = nn.Sequential(
            nn.Linear(1, n_clusters),
            nn.Softmax(dim=1)
        )

        self.dropout = nn.Dropout(p=0.5)
        
        ### Sentiment Classification model layers ###
        self.linear_basic_betw_2 = nn.Sequential(
          nn.Linear(inter_layer_size, inter_layer_size),
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

    def forward(self, cluster_number, inputs_list=[]):
        output_list = []
        for i in range(n_clusters):
          normalized_data = inputs_list[:,i]
          output_list.append(normalized_data)

        concat_emb = torch.cat(output_list, dim=-1)
        embbedding_layer = self.linear_basic_betw(concat_emb)
        ### foward sentiment classification ###
        x_sent = self.linear_basic_betw_2(embbedding_layer)
        sent_output = self.fc_class(x_sent)
        ###

        ### Embedding reconstruction ###
        x_reconstruction = self.reconstruction_layer_1(embbedding_layer)
        reconstruction_output = self.reconstruction_layer_2(x_reconstruction)
        return sent_output, reconstruction_output, embbedding_layer

emb_dim = len(df_train.iloc[0]['cluster_0'])
print(f"embedding dim : {emb_dim}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ClassificationModel(emb_dim, inter_layer_size=64, n_classes=3, n_domains=6, alpha=domain_alpha)
model = model.to(device)

criterion_sent = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()
reconstruction_criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

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

def evaluate_dataset(cur_data_loader, cur_split="test"):
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

          optimizer.zero_grad()

          sent_outputs, _, _ = model(cur_cluster_label, cur_list_embeddings)

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
  return val_f1[0], y_pred, list_all_indices, val_loss

best_f1_score = 0
best_predictions = []
cur_training_pacience = 0
best_val_loss = float('inf')

for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0.0
    for cur_list_embeddings, cur_cluster_label, cur_sentiment_label, cur_index, cur_domain in train_dataloader:
        l_indexes = list(cur_index)
        cur_list_embeddings, cur_cluster_label = cur_list_embeddings.to(device), cur_cluster_label.to(device)
        cur_sentiment_label = cur_sentiment_label.to(device)
        cur_domain = cur_domain.to(device)


        optimizer.zero_grad()
        sent_outputs, reconstruction_output, embbedding_layer = model(cur_cluster_label, cur_list_embeddings)
        sent_loss = criterion_sent(sent_outputs, cur_sentiment_label)
        reconstruction_loss = reconstruction_criterion(reconstruction_output, embbedding_layer)

        loss = sent_loss + reconstruction_loss


        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'\n Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader)}')

    val_f1_score, cur_y_pred, list_all_indices, cur_val_loss = evaluate_dataset(val_dataloader)
    if val_f1_score > best_f1_score:
      best_f1_score = val_f1_score
      print(f"updating f1 best score {best_f1_score}")
      torch.save(model.state_dict(), f'{output_model_path}/cluster_fusion_model_cluster_att.pt')
      best_predictions = cur_y_pred
    print(f"Current val Loss : {cur_val_loss}")

    if best_val_loss > cur_val_loss:
      best_val_loss = cur_val_loss
      print(f"Updating best validation loss : {cur_val_loss}")
      cur_training_pacience = 0
    else:
      cur_training_pacience += 1
    if cur_training_pacience > fusion_pacience:
      print("Early stopping triggered !!")
      break


    print("\n")
    print("=" * 30)
    print("\n")

model.load_state_dict(torch.load(f'{output_model_path}/cluster_fusion_model_cluster_att.pt'))

val_f1_score, cur_y_pred, list_all_indices, _ = evaluate_dataset(test_dataloader)

print(val_f1_score)

df_test['prediction'] = cur_y_pred
df_test['pred_indexes'] = list_all_indices

assert all(i==j for i, j in zip(df_test['index'], df_test['pred_indexes']))

df_test[['index', 'prediction', "pred_indexes"]].to_pickle(f'{save_path}/cluster_fusion_predictions_simple.pkl')