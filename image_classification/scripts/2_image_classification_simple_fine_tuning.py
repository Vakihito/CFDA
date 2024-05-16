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

df_train = pd.read_pickle(f"{save_path}/train_data.pkl")
df_val = pd.read_pickle(f"{save_path}/val_data.pkl")
df_test = pd.read_pickle(f"{save_path}/test_data.pkl")

batch_size = 64
num_epochs = int(os.environ['base_training_epochs'])
learning_rate = float(os.environ['base_training_lr'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the model
resnet50_model = models.resnet50(pretrained=True)

transform = transforms.Compose([
    transforms.ToTensor()
])

import torch
import torch.nn as nn

class ClassificationModel(nn.Module):
    def __init__(self, base_model, inter_layer_size=512, n_classes=3):
      super(ClassificationModel, self).__init__()
      self.base_model = base_model
      self.base_model.fc =  nn.Sequential(
        nn.Linear(self.base_model.fc.in_features, inter_layer_size),
        nn.PReLU()
      )
      self.norm_layer = nn.LayerNorm(inter_layer_size)
      # for each layer cluster embedding return the attention weights
      self.classification = nn.Sequential(
          nn.Linear(inter_layer_size, n_classes)
      )
    def forward(self, x):
      x = self.base_model(x)
      x = self.norm_layer(x)
      x = self.classification(x)
      return x

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]['data_path']
        index_data = self.dataframe.iloc[idx]['index']
        sentiment = self.dataframe.iloc[idx]['sentiment']
        image_path = f"{images_path}/{image_path}"
        # Load image
        img = Image.open(image_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, sentiment, index_data

g = torch.Generator()
g.manual_seed(0)
train_dataset = CustomDataset(dataframe=df_train, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, generator=g)

test_dataset = CustomDataset(dataframe=df_test, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=2)

val_dataset = CustomDataset(dataframe=df_val, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=2)

model = ClassificationModel(resnet50_model)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
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
      for images, labels, index in cur_data_loader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          output_softmax = softmax_function(outputs.data)
          _, predicted = torch.max(output_softmax, 1)

          list_labels.append(labels.detach().cpu().numpy())
          list_confidence.append(output_softmax.detach().cpu().numpy())
          list_predictions.append(predicted.detach().cpu().numpy())
          list_index += index

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

for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0.0

    for images, labels, _ in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader)}')

    val_f1_score, cur_y_pred, _, _ = evaluate_dataset(val_dataloader)
    if val_f1_score > best_f1_score:
      best_f1_score = val_f1_score
      print(f"updating f1 best score {best_f1_score}")
      torch.save(model.state_dict(), f'{output_model_path}/simple_finetuning.pt')

model.load_state_dict(torch.load(f'{output_model_path}/simple_finetuning.pt'))

val_f1_score, cur_y_pred, indexes_data, cur_y_confidence = evaluate_dataset(test_dataloader)

df_test['prediction'] = cur_y_pred
df_test['confidence'] = cur_y_confidence
df_test['pred_indexes'] = indexes_data

assert all(i==j for i, j in zip(df_test['index'], df_test['pred_indexes']))

df_test[['index', 'prediction', 'confidence', 'pred_indexes']].to_pickle(f'{save_path}/simple_predictions.pkl')