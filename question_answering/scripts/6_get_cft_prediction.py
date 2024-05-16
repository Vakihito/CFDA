print("###############################")
print("##### Starting notebook 6 #####")
print("###############################")

from torch.utils.data import DataLoader
import os
import shutil
import pandas as pd
import torch
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
import os
import shutil
import pandas as pd

max_length = 512  # The maximum length of a feature (question and context)
doc_stride = 128  # The allowed overlap between two part of the context when splitting is performed.

n_clusters = int(os.environ["n_clusters"])
read_data_path = os.environ["read_data_path"]
base_model=os.environ["base_model"]

emsamble_quantile = int(os.environ["emsamble_quantile"])

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
  map_id_datasets = {cur_id: cur_dataset for cur_id, cur_dataset in zip(ids, datasets)}

  df_to_match["context"] = df_to_match["index"].apply(lambda x: map_id_context[x])
  df_to_match["question"] = df_to_match["index"].apply(lambda x: map_id_questions[x])
  df_to_match["cluster_labels"]  = df_to_match["index"].apply(lambda x: map_id_clusters[x])
  df_to_match["dataset"]  = df_to_match["index"].apply(lambda x: map_id_datasets[x])
  df_to_match["id"]  = df_to_match["index"].apply(lambda x: x)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(base_model)
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

df_test = pd.read_pickle(f"{read_data_path}/test_data_clustered.pkl")

important_cols = ["context",	"question",	"cluster_labels",
                  "dataset",	"reponse_str",
                  "start_prob",	"end_prob",	"mean_prob",
                  "start_pos",	"end_post", "id",
                  "start_prob_all_tokens", "end_prob_all_tokens"]

# getting the cluster predictions
list_cluster_preds_test = []
for cur_cluster in range(n_clusters):
  cur_df_test_preds = pd.read_pickle(f"{os.environ['read_data_path']}/test_data_with_cluster_preds_{cur_cluster}.pkl")
  list_cluster_preds_test.append(cur_df_test_preds)

df_test_with_cluster_preds = pd.concat(list_cluster_preds_test)
df_test_cluster_predictions = get_final_prediction(df_test_with_cluster_preds, df_test)

# getting the base predictions
df_test_base_preds = pd.read_pickle(f"{os.environ['read_data_path']}/test_data_with_base_preds.pkl")
df_test_base_predictions = get_final_prediction(df_test_base_preds, df_test)

df_test.sort_values(by='id',inplace=True)
df_test_base_predictions.sort_values(by='id',inplace=True)
df_test_cluster_predictions.sort_values(by='id',inplace=True)


assert all(df_test_base_predictions['id'].values == df_test['id'].values)
assert all(df_test_cluster_predictions['id'].values == df_test['id'].values)

df_test_base_predictions.to_pickle(f"{os.environ['read_data_path']}/base_formated_predictions.pkl")
df_test_cluster_predictions.to_pickle(f"{os.environ['read_data_path']}/cluster_formated_predictions.pkl")

"""# Getting the metrics"""

import os

dataset_name = os.environ["dataset_test"]
model_name = os.environ["base_model"]
data_path = os.environ['read_data_path']
number_clusters = int(os.environ["n_clusters"])

df_test['base_prediction'] = df_test_base_predictions['reponse_str'].values
df_test['cluster_prediction'] = df_test_cluster_predictions['reponse_str'].values

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

normal_pred = get_metrics_for_df(df_test, model_name, "base_prediction")
cluster_pred = get_metrics_for_df(df_test, model_name, "cluster_prediction")
cst_pred = pd.read_pickle(f"{read_data_path}/cst_results.pkl")

results = pd.concat([normal_pred, cluster_pred, cst_pred])

results

results.to_pickle(f"{data_path}/results.pkl")