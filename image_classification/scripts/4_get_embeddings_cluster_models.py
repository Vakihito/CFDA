import os

number_clusters = int(os.environ['n_clusters'])
base_path = os.environ['base_path']
models_path = os.environ['models_path'] 
cur_cluster_number = os.environ["cur_cluster_number"]

random_state = int(os.environ['random_state'])


data_path_train = os.environ['s4_data_path_train']
data_path_test = os.environ['s4_data_path_test']
data_path_val = f"{data_path_test[:-4]}_val.pkl"

data_path_train_cts_emb = os.environ['s4_data_path_train_cts_emb']
data_path_test_cts_emb = os.environ['s4_data_path_test_cts_emb']

if int(cur_cluster_number) > 0:
  data_path_train = data_path_train_cts_emb
  data_path_test = data_path_test_cts_emb
  data_path_val = f"{data_path_test_cts_emb[:-4]}_val.pkl"

from datasets import load_dataset
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import torch

import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm

random.seed(random_state)
tf.random.set_seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)

import ktrain

df_train = pd.read_pickle(data_path_train)
df_val = pd.read_pickle(data_path_val)
df_test = pd.read_pickle(data_path_test)

class DistilBertEmbeddings(tf.keras.layers.Layer):
    def __init__(self, cur_predictor):
        super(DistilBertEmbeddings, self).__init__()
        self.distilbert = cur_predictor.model.layers[0]

    def call(self, inputs):
        outputs = self.distilbert(inputs)
        del inputs
        return outputs[0]

def get_embeddings(cur_df, cur_cluster):
  batch_size = 4
  all_texts = list(cur_df['text'].values.tolist())
  list_all_embeddings = []
  print("getting embeddings")
  for i in tqdm(range(0, len(all_texts), batch_size), total=len(all_texts)//batch_size):
    texts = all_texts[i: i + batch_size]
    encoded_inputs = tokenizer.batch_encode_plus(texts, padding=True, return_tensors='tf')
    embeddings = distilbert_embeddings(encoded_inputs).numpy()
    for cur_embedding in embeddings:
      list_all_embeddings.append(np.copy(cur_embedding[0]))
    del encoded_inputs
    del embeddings
  cur_df[f'embeddings_cluster_{cur_cluster}'] = list_all_embeddings
  del list_all_embeddings


print("getting the predictor")
predictor = ktrain.load_predictor(f"{models_path}/finetuned_cluster_{cur_cluster_number}")
print("loaded the predictor")

distilbert_embeddings = DistilBertEmbeddings(predictor)
tokenizer = predictor.preproc.get_tokenizer()
get_embeddings(df_train, cur_cluster_number)
get_embeddings(df_test, cur_cluster_number)
get_embeddings(df_val,cur_cluster_number)
print(f"done cluster {cur_cluster_number}!")
del distilbert_embeddings
del tokenizer
del predictor

df_train.to_pickle(data_path_train_cts_emb)
df_test.to_pickle(data_path_test_cts_emb)
df_val.to_pickle(f"{data_path_test_cts_emb[:-4]}_val.pkl")