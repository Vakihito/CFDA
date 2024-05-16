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

save_path = os.environ['save_path']
images_path = os.environ["read_data_path"]
test_dataset = os.environ["dataset_test"]
n_clusters = os.environ["n_clusters"]
output_model_path = os.environ['output_model_path']

cur_cluster = int(os.environ['cur_cluster'])

df_train = pd.read_pickle(f"{save_path}/train_data_clustered.pkl")
df_val = pd.read_pickle(f"{save_path}/val_data_clustered.pkl")
df_test = pd.read_pickle(f"{save_path}/test_data_clustered.pkl")

df_train.head(3)

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

df_train = df_train[df_train['cluster_labels'] == cur_cluster].reset_index(drop=True)
df_val = df_val[df_val['cluster_labels'] == cur_cluster].reset_index(drop=True)
df_test = df_test[df_test['cluster_labels'] == cur_cluster].reset_index(drop=True)

batch_size = 64
num_epochs = int(os.environ['base_training_epochs'])
learning_rate = float(os.environ['base_training_lr'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the model
resnet50_model = models.resnet50(pretrained=True)

transform = transforms.Compose([
    transforms.ToTensor()
])

class ReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

import torch
import torch.nn as nn

class ClassificationModel(nn.Module):
    def __init__(self, base_model, inter_layer_size=512, n_classes=3, n_domains=6):
      super(ClassificationModel, self).__init__()
      self.base_model = base_model

      self.base_model.fc =  nn.Sequential(
        nn.Linear(self.base_model.fc.in_features, inter_layer_size),
        nn.PReLU()
      )
      self.norm_layer = nn.LayerNorm(inter_layer_size)
      # classification layers for sentiment classification #
      self.classification = nn.Sequential(
          nn.Linear(inter_layer_size, n_classes)
      )

      # classification layers for domain classification #
      self.domain_classification = nn.Sequential(
          nn.Linear(inter_layer_size, n_domains)
      )
      
      # embedding reconstrution layer #
      self.reconstruction_layer_1 = nn.Sequential(
        nn.Linear(inter_layer_size, inter_layer_size // 4),
        nn.PReLU()
      )

      self.reconstruction_layer_2 = nn.Sequential(
        nn.Linear(inter_layer_size // 4, inter_layer_size),
        nn.PReLU()
      )


    def forward(self, x, alpha):
      embedding_features = self.base_model(x)
      norm_embedding_features = self.norm_layer(embedding_features)

      reversed_norm_embedding_features = ReverseLayer.apply(norm_embedding_features, alpha)
      # sentiment classification
      sent_output = self.classification(norm_embedding_features)
      # domain classification
      domain_output = self.domain_classification(reversed_norm_embedding_features)
      # auto encoder embedding recreation
      x_reconstruction = self.reconstruction_layer_1(norm_embedding_features)
      reconstruction_output = self.reconstruction_layer_2(x_reconstruction)
      
      return sent_output, domain_output, reconstruction_output, norm_embedding_features

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.list_img_path = self.dataframe["data_path"].values
        self.list_idx = self.dataframe["index"].values
        self.list_sent = self.dataframe["sentiment"].values
        self.list_domain = self.dataframe["domain"].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.list_img_path[idx]
        index_data = self.list_idx[idx]
        sentiment = self.list_sent[idx]
        cur_domain = self.list_domain[idx]

        image_path = f"{images_path}/{image_path}"
        # Load image
        img = Image.open(image_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, sentiment, index_data, cur_domain

g = torch.Generator()
g.manual_seed(0)

train_dataset = CustomDataset(dataframe=df_train, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=2, generator=g)

test_dataset = CustomDataset(dataframe=df_test, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=2)

val_dataset = CustomDataset(dataframe=df_val, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=2)

model = ClassificationModel(resnet50_model, n_domains=domain_counter)
model = model.to(device)

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.CrossEntropyLoss()
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

import numpy as np

softmax_function = nn.Softmax(dim=-1)
def evaluate_dataset(cur_data_loader):
  model.eval()
  correct = 0
  total = 0
  list_labels = []
  list_predictions = []
  list_index = []
  list_confidence = []

  with torch.no_grad():
      for images, labels, indexes, _ in cur_data_loader:
          images, labels = images.to(device), labels.to(device)
          outputs, _ , _, _ = model(images, 0.0)
          output_softmax = softmax_function(outputs.data)
          _, predicted = torch.max(output_softmax, 1)

          list_labels.append(labels.detach().cpu().numpy())
          list_confidence.append(output_softmax.detach().cpu().numpy())
          list_predictions.append(predicted.detach().cpu().numpy())
          list_index += indexes

  y_true = np.concatenate(list_labels).tolist()
  y_pred = np.concatenate(list_predictions).tolist()
  y_confidence = np.concatenate(list_confidence).tolist()
  prediction_metrics = validation_metrics(y_true, y_pred)
  val_f1 = prediction_metrics['f1-macro']
  print("\n")
  print("Test Metrics")
  for cur_metric in prediction_metrics:
    print(f"{cur_metric} : {prediction_metrics[cur_metric]}")
  print("#" * 20)
  return val_f1[0], y_pred, list_index, y_confidence

best_f1_score = 0
best_predictions = []
len_dataloader = len(train_dataloader)

for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0.0
    i = 0
    for images, class_labels, _, domain_labels in train_dataloader:
        p = (float(i + epoch * len_dataloader) / num_epochs) / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        images, class_labels, domain_labels = images.to(device), class_labels.to(device), domain_labels.to(device)
        optimizer.zero_grad()
        class_outputs, domain_outputs, emb_reconstruction, emb_target = model(images, alpha)


        class_loss = class_criterion(class_outputs, class_labels)
        reconstruction_loss = reconstruction_criterion(emb_reconstruction, emb_target)
        loss = class_loss + reconstruction_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        i += 1
        
    print("\n Training loss : \n")
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader)}')

    val_f1_score, cur_y_pred, list_index, y_confidence = evaluate_dataset(val_dataloader)
    if val_f1_score > best_f1_score:
      best_f1_score = val_f1_score
      print(f"updating f1 best score {best_f1_score}")
      torch.save(model.state_dict(), f'{output_model_path}/cluster_model_{cur_cluster}.pt')
      best_predictions = cur_y_pred

model.load_state_dict(torch.load(f'{output_model_path}/cluster_model_{cur_cluster}.pt'))

val_f1_score, cur_y_pred, list_index, cur_y_confidence = evaluate_dataset(test_dataloader)

df_test['prediction'] = cur_y_pred
df_test['confidence'] = cur_y_confidence
df_test['pred_indexes'] = list_index

assert all(i==j for i, j in zip(df_test['index'], df_test['pred_indexes']))

df_test[['index', 'prediction', 'confidence', 'pred_indexes']].to_pickle(f'{save_path}/cluster_{cur_cluster}_predictions.pkl')