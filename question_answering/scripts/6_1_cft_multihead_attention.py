from torch.utils.data import DataLoader
import os
import shutil

"""### Notebook 5

---
"""

n_clusters = int(os.environ["n_clusters"])

from datasets import load_dataset, Dataset
from datasets import ClassLabel, Sequence
import torch
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

## sentiment models parameters ##
model_checkpoint = os.environ["base_model"]
read_data_path = os.environ["read_data_path"]
dataset_name = os.environ["dataset_test"]
encoder_layers_to_freeze = int(os.environ["number_layers_freeze"])

num_train_epochs = int(os.environ["base_num_train_epochs"])
output_model_path = f'{os.environ["model_path"]}'

batch_size = 80

# tokenizer
max_length = 512  # The maximum length of a feature (question and context)
doc_stride = 128  # The allowed overlap between two part of the context when splitting is performed.

# hyper-parameters
learning_rate = 5e-5
weight_decay = 1e-5
random_seed = 0

df_train = pd.read_pickle(f"{read_data_path}/train_data_emsambled.pkl")
df_test = pd.read_pickle(f"{read_data_path}/test_data_clustered.pkl")
df_val = pd.read_pickle(f"{read_data_path}/val_data_clustered.pkl")

print(df_train['cluster_labels'].value_counts())

print(df_test['cluster_labels'].value_counts())

print(df_val['cluster_labels'].value_counts())

df_train.head(3)

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

train_loader = load_dataset_from_df(df_train, shuffle=True)
val_loader = load_dataset_from_df(df_val)
test_loader = load_dataset_from_df(df_test)

import torch
import torch.nn as nn


class QAModel(nn.Module):
    def __init__(self, base_model_list):
        super(QAModel, self).__init__()
        self.base_model_list = base_model_list
        self.base_model_hidden_size = base_model_list[0].config.hidden_size
        self.default_emb_size = tokenizer.model_max_length
        self.concat_size = (n_clusters + 1) * self.default_emb_size

        self.linear_basic_betw = nn.Sequential(
          nn.Linear(self.base_model_hidden_size, self.default_emb_size),
          nn.PReLU()
        )

        self.linear_1 = nn.Sequential(
          nn.Linear(self.concat_size, self.default_emb_size),
          nn.PReLU()
        )

        self.linear_2 = nn.Sequential(
          nn.Linear(self.default_emb_size, self.default_emb_size),
          nn.PReLU()
        )

        self.single_dense = nn.Sequential(
            nn.Linear(self.default_emb_size, self.default_emb_size),
            nn.PReLU()
        )

        self.fc_positions = nn.Sequential(
            nn.Linear(self.default_emb_size, 2),
        )
        self.att_layer = nn.MultiheadAttention(self.concat_size, num_heads=16, batch_first=True)


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

    def forward(self, input_ids, attention_mask):
        list_embeddings = []
        output_list = []
        for cur_base_model in self.base_model_list:
          x = cur_base_model(input_ids=input_ids, attention_mask=attention_mask)
          x = x.last_hidden_state
          embbedding_layer = self.linear_basic_betw(x)
          list_embeddings.append(embbedding_layer)

        embeddings_out = torch.cat(list_embeddings, dim=2)
        attn_output, _ = self.att_layer(embeddings_out, embeddings_out, embeddings_out)

        embbedding_layer_cat = (attn_output * embeddings_out + embeddings_out)
        emb_linear_1 = self.linear_1(embbedding_layer_cat)
        x = self.linear_2(emb_linear_1)
        final_embedding = x + emb_linear_1
        x = self.fc_positions(final_embedding)

        ####################################
        ##### foward reconstruction #####
        ####################################
        x_reconstruction = self.reconstruction_layer_1(final_embedding)

        reconstruction_output = self.reconstruction_layer_2(x_reconstruction)

        return x[:,:,0], x[:,:,1], final_embedding, reconstruction_output

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
        start_logits, end_logits, embbedding_layer, _ = qa_model(**dict_inputs)

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

input_columns = set(['input_ids', 'attention_mask'])
softmax_func = nn.Softmax(dim=1)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, qa_model.parameters()), lr=learning_rate)

best_validation_loss = float('inf')
epochs_without_improvement = 0
loss_function = nn.CrossEntropyLoss()
criterion_reconstruction = nn.MSELoss()

#####################
### training loop ###
#####################
list_positions_s = []
list_positions_e = []
with tqdm(total=num_train_epochs * len(train_loader)) as pbar:
  for epoch in range(num_train_epochs):
    total_loss_train = 0
    for cur_sample in train_loader:
      qa_model.train()
      optimizer.zero_grad()
      dict_inputs = {} # creating dict to be used as input
      # seding data to device
      for cur_inputs in cur_sample:
        cur_sample[cur_inputs] = cur_sample[cur_inputs].to(device)
        if cur_inputs in input_columns:
          dict_inputs[cur_inputs] = cur_sample[cur_inputs]

      # Forward pass
      start_logits, end_logits, embbedding_layer, reconstruction_output = qa_model(**dict_inputs)

      # getting the loss out of the start and end positions #
      loss_1 = loss_function(start_logits, cur_sample['start_positions'])
      loss_2 = loss_function(end_logits, cur_sample['end_positions'])

      list_positions_s += list(cur_sample['start_positions'].cpu().detach().numpy())
      list_positions_e += list(cur_sample['end_positions'].cpu().detach().numpy())
      loss_reconstruction = criterion_reconstruction(embbedding_layer, reconstruction_output)
      ###

      total_loss = loss_1 + loss_2 + (loss_reconstruction * 0.5)
      total_loss.backward()
      optimizer.step()
      total_loss_train += total_loss.item()
      pbar.update(1)

    total_loss_train /= len(train_loader)
    print(f"\n Epoch [{epoch+1}/{num_train_epochs}], Train Loss: {total_loss_train:.4f}")

    validation_predictions = get_prediction_from_dataset(val_loader)
    validation_loss = validation_predictions['dataset_loss']
    print(f"\n Epoch [{epoch+1}/{num_train_epochs}], Validation Loss: {validation_loss:.4f}")

    # saving the best model
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        epochs_without_improvement = 0
        torch.save(qa_model.state_dict(), f'{output_model_path}/cluster_fusion_multihead.pth')
        print(f"\n saving best_model epoch : {epoch+1}")

print("[INFO] training Step Completed !")

# loading the best model
qa_model.load_state_dict(torch.load(f'{output_model_path}/cluster_fusion_multihead.pth'))

"""## Getting the prediction of all the datasets

---
"""

train_loader = load_dataset_from_df(df_train)
val_loader = load_dataset_from_df(df_val)
test_loader = load_dataset_from_df(df_test)

test_predictions = get_prediction_from_dataset(test_loader)
df_test_predictions = pd.DataFrame(test_predictions)

df_test_predictions.to_pickle(f"{os.environ['read_data_path']}/test_data_with_preds_cst_multihead.pkl")