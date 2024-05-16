print("###############################")
print("##### Starting notebook 5 #####")
print("###############################")

import os
import shutil
from torch.utils.data import DataLoader
import os
import shutil
from datasets import load_dataset, Dataset
from datasets import ClassLabel, Sequence
import torch
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

test_dataset = os.environ["dataset_test"]
prediction_type = "cst_mhead"
n_clusters = int(os.environ["n_clusters"])

model_checkpoint = os.environ["base_model"]
read_data_path = os.environ["read_data_path"]
dataset_name = os.environ["dataset_test"]
num_train_epochs = int(os.environ["base_num_train_epochs"])
encoder_layers_to_freeze = int(os.environ["number_layers_freeze"])
output_model_path = f'{os.environ["model_path"]}'

batch_size = 80

# tokenizer
max_length = 512  # The maximum length of a feature (question and context)
doc_stride = 128  # The allowed overlap between two part of the context when splitting is performed.

# hyper-parameters
learning_rate = 1e-4
weight_decay = 1e-5
random_seed = 0

from collections import defaultdict
def create_split_count(cur_df):
  split_count = defaultdict(lambda : 0)
  list_splits = []
  for cur_indx in cur_df['index']:
    list_splits.append(split_count[cur_indx])
    split_count[cur_indx] += 1
  return list_splits


def match_idx_to_name(df_to_match, df_with_names):
  ids = df_with_names["id"].values
  contexts = df_with_names["context"].values
  questions = df_with_names["question"].values
  clusters = df_with_names["cluster_labels"].values
  datasets = df_with_names['dataset'].values

  map_id_context = {cur_id: cur_context for cur_id, cur_context in zip(ids, contexts)}
  map_id_questions = {cur_id: cur_question for cur_id, cur_question in zip(ids, questions)}
  map_id_clusters = {cur_id: cur_clusters for cur_id, cur_clusters in zip(ids, clusters)}
  map_id_datasets = {cur_id: cur_dataset for cur_id, cur_dataset in zip(ids, datasets)}

  df_to_match["context"] = df_to_match["index"].apply(lambda x: map_id_context[x])
  df_to_match["question"] = df_to_match["index"].apply(lambda x: map_id_questions[x])
  df_to_match["cluster_labels"]  = df_to_match["index"].apply(lambda x: map_id_clusters[x])
  df_to_match["dataset"]  = df_to_match["index"].apply(lambda x: map_id_datasets[x])
  df_to_match["id"]  = df_to_match["index"].apply(lambda x: x)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
pad_on_right = tokenizer.padding_side == "right"

def get_single_prediction_text(cur_start, cur_end, cur_context, cur_question, sample_idx):
  tokenized_examples = tokenizer(
        [cur_question if pad_on_right else cur_context],
        [cur_context if pad_on_right else cur_question],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
  start_offset = tokenized_examples.encodings[sample_idx].offsets[cur_start][0]
  end_offset = tokenized_examples.encodings[sample_idx].offsets[cur_end][1]
  return cur_context[start_offset:end_offset]

def prediction_function(cur_df):
  idx_start = cur_df["start_positions"].values
  idx_end = cur_df["end_positions"].values
  list_input_ids = cur_df["input_ids"].values
  list_contexts = cur_df["context"].values
  list_questions = cur_df["question"].values
  list_split = cur_df['split'].values

  idx_start_prob = cur_df["prob_ini"].values
  idx_end_prob = cur_df["prob_end"].values

  dict_responses = {
      "reponse_str" : [],
      "start_prob" : [],
      "start_prob_all_tokens" : [],
      "end_prob" : [],
      "end_prob_all_tokens" : [],
      "mean_prob" : [],
      "start_pos": [],
      "end_post": []
  }


  for i, cur_start, cur_end, cur_context, cur_question, cur_split in zip(range(len(idx_start)) ,idx_start, idx_end, list_contexts, list_questions, list_split):
    cur_input_ids = list_input_ids[i]
    cur_start_idx_prob = idx_start_prob[i]
    cur_end_idx_prob = idx_end_prob[i]

    cur_start_idx_prob_max = np.max(idx_start_prob[i])
    cur_end_idx_prob_max = np.max(idx_end_prob[i])


    # cases of empty response
    if (cur_start <= 0) or (cur_end <= 0) or (cur_start + 1 > len(cur_input_ids)) or ((cur_end + 1 > len(cur_input_ids))):
      dict_responses["reponse_str"].append("")
      dict_responses["start_prob_all_tokens"].append(cur_start_idx_prob)
      dict_responses["end_prob_all_tokens"].append(cur_end_idx_prob)
      dict_responses["start_prob"].append(0.)
      dict_responses["end_prob"].append(0.)
      dict_responses["mean_prob"].append(0.)
      dict_responses["start_pos"].append(cur_start)
      dict_responses["end_post"].append(cur_end)
    else:
      if (cur_start > cur_end + 1):
        cur_end = cur_start + 1

      answer_prob = (cur_start_idx_prob_max + cur_end_idx_prob_max) / 2
      convert_tokens = lambda x: tokenizer.convert_tokens_to_string((tokenizer.convert_ids_to_tokens(x)))
      prediction_text = get_single_prediction_text(cur_start, cur_end, cur_context, cur_question, cur_split)
      dict_responses["reponse_str"].append(prediction_text)
      dict_responses["start_prob_all_tokens"].append(cur_start_idx_prob)
      dict_responses["end_prob_all_tokens"].append(cur_end_idx_prob)
      dict_responses["start_prob"].append(cur_start_idx_prob_max)
      dict_responses["end_prob"].append(cur_end_idx_prob_max)
      dict_responses["mean_prob"].append(answer_prob)
      dict_responses["start_pos"].append(cur_start)
      dict_responses["end_post"].append(cur_end)


  return dict_responses

def get_single_prediction(cur_df_sampled):
  mean_probs = cur_df_sampled["mean_prob"].values
  idx = np.argmax(mean_probs)
  return cur_df_sampled.iloc[idx]


def get_final_prediction(cur_df, cur_df_base):
  cur_df["split"] = create_split_count(cur_df)
  match_idx_to_name(cur_df, cur_df_base)

  dict_responses = prediction_function(cur_df)
  for cur_key in dict_responses:
    cur_df[cur_key] = dict_responses[cur_key]

  final_predictions = []
  all_indexes = cur_df["index"].unique()
  for cur_idx in all_indexes:
    cur_df_sampled = cur_df[cur_df["index"] == cur_idx]
    final_predictions.append(get_single_prediction(cur_df_sampled))
  return pd.DataFrame(final_predictions)

df_train = pd.read_pickle(f"{read_data_path}/train_data_emsambled.pkl")
df_val = pd.read_pickle(f"{read_data_path}/val_data_clustered.pkl")
df_test = pd.read_pickle(f"{read_data_path}/test_data_clustered.pkl")

"""## Finetuning Models"""

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
pad_on_right = tokenizer.padding_side == "right"

def prepare_train_features(examples):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["sample_idx"] = []
    tokenized_examples["cluster"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        cur_id = examples["id"][sample_index]
        cur_cluster = examples["cluster_labels"][sample_index]

        tokenized_examples["cluster"].append(cur_cluster)
        tokenized_examples["sample_idx"].append(cur_id)
        try:
          # If no answers are given, set the cls_index as answer.
          if len(answers["answer_start"]) == 0:
              tokenized_examples["start_positions"].append(cls_index)
              tokenized_examples["end_positions"].append(cls_index)
          else:
              # Start/end character index of the answer in the text.
              start_char = answers["answer_start"][0]
              end_char = start_char + len(answers["text"][0])

              # Start token index of the current span in the text.
              token_start_index = 0
              while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                  token_start_index += 1

              # End token index of the current span in the text.
              token_end_index = len(input_ids) - 1
              while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                  token_end_index -= 1

              # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
              if not (
                  offsets[token_start_index][0] <= start_char
                  and offsets[token_end_index][1] >= end_char
              ):
                  tokenized_examples["start_positions"].append(cls_index)
                  tokenized_examples["end_positions"].append(cls_index)
              else:
                  # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                  # Note: we could go after the last offset if the answer is the last word (edge case).
                  while (
                      token_start_index < len(offsets)
                      and offsets[token_start_index][0] <= start_char
                  ):
                      token_start_index += 1
                  tokenized_examples["start_positions"].append(token_start_index - 1)
                  while offsets[token_end_index][1] >= end_char:
                      token_end_index -= 1
                  tokenized_examples["end_positions"].append(token_end_index + 1)
        except:
          print(f"cur_id : {cur_id}")
    return tokenized_examples

def load_dataset_from_df(cur_df, shuffle=False):
  datasets = Dataset.from_pandas(cur_df)
  tokenized_datasets = datasets.map(
    prepare_train_features, batched=True, remove_columns=datasets.column_names
  )
  tokenized_datasets.set_format("torch")
  if shuffle == False:
    cur_dataloader = DataLoader(
      tokenized_datasets,
      shuffle=shuffle,
      batch_size=batch_size,
    )
  else:
    g = torch.Generator()
    g.manual_seed(0)
    cur_dataloader = DataLoader(
      tokenized_datasets,
      shuffle=shuffle,
      batch_size=batch_size,
      generator=g
    )

  return cur_dataloader

from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

transformer_model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
transformer_model = transformer_model.to(device)

df_target = df_test[~df_test['id'].isin(df_train['id'])].reset_index(drop=True)

train_loader = load_dataset_from_df(df_train, shuffle=True)
target_loader = load_dataset_from_df(df_target, shuffle=True)
val_loader = load_dataset_from_df(df_val)
test_loader = load_dataset_from_df(df_test)

import torch
import torch.nn as nn

class ReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class QAModel(nn.Module):
    def __init__(self, base_model_list):
        super(QAModel, self).__init__()
        self.base_model_list = base_model_list
        self.base_model_hidden_size = int(base_model_list[0].config.hidden_size)
        self.default_emb_size = 512
        self.concat_size = (n_clusters + 1) * self.default_emb_size
        self.max_length = max_length
        print("self.base_model_hidden_size : ", self.base_model_hidden_size)
        print("self.default_emb_size : ", self.default_emb_size)

        self.linear_basic_betw = nn.Sequential(
          nn.Linear(self.base_model_hidden_size, self.default_emb_size),
          nn.BatchNorm1d(self.default_emb_size),
          nn.PReLU()
        )

        self.linear_1 = nn.Sequential(
          nn.Linear(self.concat_size, self.default_emb_size),
          nn.BatchNorm1d(self.max_length),
          nn.PReLU()
        )

        self.linear_2 = nn.Sequential(
          nn.Linear(self.default_emb_size, self.default_emb_size),
          nn.BatchNorm1d(self.default_emb_size),
          nn.PReLU()
        )

        self.linear_3 = nn.Sequential(
          nn.Linear(self.default_emb_size, self.default_emb_size),
          nn.BatchNorm1d(self.default_emb_size),
          nn.PReLU()
        )

        self.single_dense = nn.Sequential(
            nn.Linear(self.default_emb_size, self.default_emb_size),
            nn.BatchNorm1d(self.default_emb_size),
            nn.PReLU()
        )

        self.fc_positions = nn.Sequential(
            nn.Linear(self.default_emb_size, 2),
        )

        self.att_layer = nn.MultiheadAttention(self.concat_size, num_heads=16, batch_first=True)

        # domain classification layer
        self.fc_domain = nn.Sequential(
            nn.Linear(self.default_emb_size, n_clusters),
        )


        #############################################
        ###### Embedding Reconstruction layers ######
        #############################################
        self.reconstruction_layer_1 = nn.Sequential(
          nn.Linear(self.default_emb_size, self.default_emb_size // 4),
          nn.PReLU()
        )

        self.reconstruction_layer_2 = nn.Sequential(
          nn.Linear(self.default_emb_size // 4, self.default_emb_size),
          nn.PReLU()
        )

    def masked_average_pooling(self, input_tensor, attention_mask):
      # Apply attention mask to the input tensor
      masked_input = input_tensor * attention_mask.unsqueeze(-1)

      # Calculate the sum of elements along the desired dimensions
      # (usually the sequence dimension)
      sum_masked_input = torch.sum(masked_input, dim=1)  # Assuming dim=1 is the sequence dimension

      # Calculate the sum of attention mask along the same dimension
      sum_attention_mask = torch.sum(attention_mask, dim=1, keepdim=True)  # Keeping the same dimensions

      # Calculate the average pooling by dividing the sum of masked input
      # by the sum of attention mask
      averaged_output = sum_masked_input / sum_attention_mask

      return averaged_output

    def forward(self, input_ids, attention_mask, alpha=0):
        list_embeddings = []
        output_list = []
        for cur_base_model in self.base_model_list:
          x = cur_base_model(input_ids=input_ids, attention_mask=attention_mask)
          x = x.last_hidden_state
          x = self.linear_basic_betw(x)
          list_embeddings.append(x)

        embeddings_out_cat = torch.cat(list_embeddings, dim=2)
        embeddings_out, att_output = self.att_layer(embeddings_out_cat, embeddings_out_cat, embeddings_out_cat)
        emb_linear_1 = self.linear_1(embeddings_out + embeddings_out_cat)
        emb_linear_2 = self.linear_2(emb_linear_1)
        emb_linear_3 = self.linear_3(emb_linear_2 + emb_linear_1)
        final_embedding = emb_linear_3 + emb_linear_2

        #################################
        ##### foward classification #####
        #################################
        x = self.fc_positions(final_embedding)


        mean_embedding = self.masked_average_pooling(final_embedding, attention_mask)
        #################################
        ##### foward reconstruction #####
        #################################
        x_reconstruction = self.reconstruction_layer_1(mean_embedding)
        reconstruction_output = self.reconstruction_layer_2(x_reconstruction)

        #################################
        ## domain classification layer ##
        #################################
        reverse_embbedding_layer = ReverseLayer.apply(mean_embedding, alpha)
        domain_output = self.fc_domain(reverse_embbedding_layer)

        return x[:,:,0], x[:,:,1], mean_embedding, reconstruction_output, domain_output

def load_model(cur_check_point):
  transformer_model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
  transformer_model = transformer_model.to(device)
  return transformer_model

def freeze_model(cur_model):
  for cur_parameters in cur_model.parameters():
    cur_parameters.requires_grad = False

def get_model(cur_checkpoint):
  transformer_model = load_model(cur_checkpoint)
  freeze_model(transformer_model)
  return transformer_model.base_model

from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

simple_base_model = get_model(f'{output_model_path}/best_model_weights_simple_finetuning.pth')
list_models = [simple_base_model]

for i in range(n_clusters):
  list_models.append(get_model(f'{output_model_path}/best_cluster_{i}_model_weights.pth'))

qa_model = QAModel(list_models)
qa_model = qa_model.to(device)

input_columns = set(['input_ids', 'attention_mask'])
softmax_func = nn.Softmax(dim=1)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, qa_model.parameters()), lr=learning_rate)

def get_prediction_from_dataset(cur_data_loader):
  # Validation phase #
  list_prob_end, list_prob_ini = ([], [])
  list_start_logits, list_end_logits = ([], [])
  list_batch_idx, list_input_ids = ([], [])

  qa_model.eval()
  dataset_loss = 0.0
  with torch.no_grad():
      for cur_sample in cur_data_loader:
        dict_inputs = {} # creating dict to be used as input
        # seding data to device
        for cur_inputs in cur_sample:
          cur_sample[cur_inputs] = cur_sample[cur_inputs].to(device)
          if cur_inputs in input_columns:
            dict_inputs[cur_inputs] = cur_sample[cur_inputs]

        # Forward pass
        start_logits, end_logits, _, _, _ = qa_model(**dict_inputs)

        # getting the loggits from the predictions
        cur_prob_dist_ini = softmax_func(start_logits.cpu().detach()).numpy()
        cur_prob_dist_end = softmax_func(end_logits.cpu().detach()).numpy()

        list_prob_ini += cur_prob_dist_ini.tolist()
        list_start_logits += np.argmax(cur_prob_dist_ini, axis=-1).tolist()

        list_prob_end += cur_prob_dist_end.tolist()
        list_end_logits += np.argmax(cur_prob_dist_end,axis=-1).tolist()
        ###

        # getting the data split back #
        list_batch_idx += [int(cur_value) for cur_value in cur_sample["sample_idx"].cpu().detach().numpy()]
        list_input_ids += list(cur_sample["input_ids"].cpu().detach().numpy())
        ###

        # getting the loss out of the start and end positions #
        loss_1 = loss_function(start_logits, cur_sample['start_positions'])
        loss_2 = loss_function(end_logits, cur_sample['end_positions'])
        ###

        cur_dataset_loss = loss_1 + loss_2
        dataset_loss += cur_dataset_loss.item()
  dataset_loss /= len(cur_data_loader)
  return {
      'dataset_loss' : dataset_loss,
      'prob_end': list_prob_end,
      'prob_ini': list_prob_ini,
      'start_positions': list_start_logits,
      'end_positions': list_end_logits,
      'index': list_batch_idx,
      'input_ids': list_input_ids
  }

best_validation_loss = float('inf')
epochs_without_improvement = 0

loss_function = nn.CrossEntropyLoss()
criterion_reconstruction = nn.MSELoss()

THETA = 0.001
GAMMA = 10 # EFFECT OF THE DOMAIN ADAPTATION LOSS 10 IS THE DEFAULT
BETHA = 0.01

len_dataloader = len(train_loader) + len(target_loader)

with tqdm(total=num_train_epochs * len(train_loader) + len(target_loader)) as pbar:
  for epoch in range(num_train_epochs):
    total_loss_train = 0
    batch_idx = 0
    for cur_sample in train_loader:
      p = (batch_idx + epoch * len_dataloader) / (num_train_epochs * len_dataloader)
      alpha  = 2. / (1. + np.exp(-GAMMA * p)) - 1

      qa_model.train()
      optimizer.zero_grad()
      dict_inputs = {} # creating dict to be used as input
      # seding data to device
      for cur_inputs in cur_sample:
        cur_sample[cur_inputs] = cur_sample[cur_inputs].to(device)
        if cur_inputs in input_columns:
          dict_inputs[cur_inputs] = cur_sample[cur_inputs]
      dict_inputs['alpha']  = alpha
      # Forward pass
      start_logits, end_logits, emb_model, emb_recuns, domain_output = qa_model(**dict_inputs)

      # getting the loss out of the start and end positions #
      loss_1 = loss_function(start_logits, cur_sample['start_positions'])
      loss_2 = loss_function(end_logits, cur_sample['end_positions'])
      loss_rec = criterion_reconstruction(emb_recuns, emb_model)
      loss_domain = loss_function(domain_output, cur_sample['cluster'])
      ###

      total_loss = loss_1 + loss_2 + (loss_rec * THETA + loss_domain * BETHA)
      total_loss.backward()
      optimizer.step()
      total_loss_train += total_loss.item()
      batch_idx += 1
      pbar.update(1)

    for cur_sample in target_loader:
      p = (batch_idx + epoch * len_dataloader) / (num_train_epochs * len_dataloader)
      alpha  = 2. / (1. + np.exp(-GAMMA * p)) - 1

      qa_model.train()
      optimizer.zero_grad()
      dict_inputs = {} # creating dict to be used as input
      # seding data to device
      for cur_inputs in cur_sample:
        cur_sample[cur_inputs] = cur_sample[cur_inputs].to(device)
        if cur_inputs in input_columns:
          dict_inputs[cur_inputs] = cur_sample[cur_inputs]
      dict_inputs['alpha'] = alpha
      # Forward pass
      _, _, emb_model, emb_recuns, domain_output= qa_model(**dict_inputs)

      # getting the loss out of the start and end positions #
      loss_rec = criterion_reconstruction(emb_recuns, emb_model)
      loss_domain = loss_function(domain_output, cur_sample['cluster'])
      ###

      total_loss = loss_rec * THETA + loss_domain * BETHA
      total_loss.backward()
      optimizer.step()
      total_loss_train += total_loss.item()
      batch_idx += 1
      pbar.update(1)

    total_loss_train /= len_dataloader
    print(f"\n Epoch [{epoch+1}/{num_train_epochs}], Train Loss: {total_loss_train:.4f}")

    validation_predictions = get_prediction_from_dataset(val_loader)
    validation_loss = validation_predictions['dataset_loss']
    print(f"\n Epoch [{epoch+1}/{num_train_epochs}], Validation Loss: {validation_loss:.4f}")


    # saving the best model
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        epochs_without_improvement = 0
        torch.save(qa_model.state_dict(), f'{output_model_path}/model_CST_mhead.pth')
        print(f"\n saving best_model epoch : {epoch+1}")

"""## Get test predictions

---
"""

state_dict = torch.load(f'{output_model_path}/model_CST_mhead.pth')
qa_model.load_state_dict(state_dict)

test_predictions = get_prediction_from_dataset(test_loader)

df_test_predictions = pd.DataFrame(test_predictions)

df_test_predictions.head(3)

important_cols = ["context",	"question",	"cluster_labels",
                  "dataset",	"reponse_str",
                  "start_prob",	"end_prob",	"mean_prob",
                  "start_pos",	"end_post", "id",
                  "start_prob_all_tokens", "end_prob_all_tokens"]

df_pseudo_pred = get_final_prediction(df_test_predictions, df_test)

df_test[prediction_type] = list(df_pseudo_pred['reponse_str'].values)

import numpy as np
import pandas as pd
from evaluate import load

squad_metric = load("squad")

def get_metrics_for_df(cur_df, model_name, prediction_col):
  formatted_predictions_squad = [
    {"id": str(k), "prediction_text": v} for k, v in cur_df[['id', prediction_col]].values
  ]
  references_squad = [
      {"id": str(k), "answers": v} for k, v in cur_df[['id', 'answers']].values
  ]

  squad_metrics_results = squad_metric.compute(predictions=formatted_predictions_squad, references=references_squad)

  dict_metrics = {
      'dataset' : dataset_name,
      'model_name' : model_name,
      'exact_match' :  [squad_metrics_results['exact_match'] / 100],
      'f1' :           [squad_metrics_results['f1'] / 100],
      'type': prediction_col
  }

  return pd.DataFrame( dict_metrics)

df_metrics_pseudo = get_metrics_for_df(df_test, model_checkpoint , prediction_type)

df_metrics_pseudo.to_pickle(f"{read_data_path}/cst_results.pkl")
