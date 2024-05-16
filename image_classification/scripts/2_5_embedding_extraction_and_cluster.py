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
n_clusters = int(os.environ["n_clusters"])
output_model_path = os.environ['output_model_path']

df_train = pd.read_pickle(f"{save_path}/train_data.pkl")
df_val = pd.read_pickle(f"{save_path}/val_data.pkl")
df_test = pd.read_pickle(f"{save_path}/test_data.pkl")

batch_size = 800

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the model

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
      return x

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        cur_image_path = self.dataframe.iloc[idx]['data_path']
        index_data = self.dataframe.iloc[idx]['index']
        cur_image_path = f"{images_path}/{cur_image_path}"
        # Load image
        img = Image.open(cur_image_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return {'image': img, 'index_data':index_data}

train_dataset = CustomDataset(dataframe=df_train, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

val_dataset = CustomDataset(dataframe=df_val, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = CustomDataset(dataframe=df_test, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

resnet50_model = models.resnet50(pretrained=True)
model = ClassificationModel(resnet50_model)


# loading the simple model #
model.load_state_dict(torch.load(f'{output_model_path}/simple_finetuning.pt'))
model = model.to(device)
###

def get_embedding_from_dataloader(cur_dataloader):
  embeddings = []

  # Loop through the dataloader
  for batch in tqdm(cur_dataloader, total=(len(cur_dataloader))):
      images = batch['image']
      images = Variable(images).to(device)

      with torch.no_grad():
          outputs = model(images)

      embeddings.append(outputs.cpu().numpy())

  embeddings = np.concatenate(embeddings)

  return list(embeddings)

df_train["embeddings"] = get_embedding_from_dataloader(train_dataloader)
df_val["embeddings"] = get_embedding_from_dataloader(val_dataloader)
df_test["embeddings"] = get_embedding_from_dataloader(test_dataloader)

k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(df_train['embeddings'].values.tolist())

## getting the cluster of each dataset
df_train['cluster_labels'] = k_means.labels_
df_test['cluster_labels'] = k_means.predict(df_test['embeddings'].values.tolist())
df_val['cluster_labels'] = k_means.predict(df_val['embeddings'].values.tolist())

print("train dataframe distribution : ")
print(df_train['cluster_labels'].value_counts())

print("test dataframe distribution : ")
print(df_test['cluster_labels'].value_counts())
###

df_train.to_pickle(f"{save_path}/train_data_clustered.pkl")
df_val.to_pickle(f"{save_path}/val_data_clustered.pkl")
df_test.to_pickle(f"{save_path}/test_data_clustered.pkl")