import numpy as np
import os
import pandas as pd
from   pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import time
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

formalism_dir = os.getcwd()
labeled_data_path = os.path.join(formalism_dir, 'labeled_data', 'final_cleaned_paragraphs.csv')
output_path = os.path.join(formalism_dir, 'results', 'llama_binary')
generated_output_path = os.path.join(output_path, 'generations')
descriptive_errors_dir = os.path.join(output_path, 'errors')

device_name = 'cuda'

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(device_name)

prompt_text = """Some paragraphs in court cases interpret statutes. In this type of paragraph, there is an analysis of a statute and a claim made about its meaning.\n\nIn the following paragraph, determine if legal interpretation occurs ("INTERPRETATION") or not ("NONE")."""
inst_prompt = f"""
<s>[INST] <<SYS>>
{prompt_text}
<</SYS>>
"""

def get_interp_type(inst_prompt, text):
    full_prompt = f"{inst_prompt}\n\"{text}\"\n\nYou must respond in a single word. Your options are \"INTERPRETATION\" or \"NONE\". What is the one word that describes this paragraph? [/INST]"
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device_name)
    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=10,
    )
    output_text = tokenizer.batch_decode(generation_output[:, input_ids.shape[1]:])[0]
    return output_text


full_df = pd.DataFrame()
interpretation_df = pd.read_csv(labeled_data_path)
interpretation_df = interpretation_df[interpretation_df['class'].notna()]


for split in range(0, 5):

  split_id_file = os.path.join(formalism_dir, 'train_test_splits', f'split_{split}')

  with open(split_id_file, 'r') as file:
      train_ids = file.read().split("\n")

  interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
  interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]


  X_test = interpretation_test_df["paragraph"].to_list()
  y_test = interpretation_test_df["class"].to_list()

  predicted_labels = [get_interp_type(inst_prompt, text).upper() for text in X_test]

  def clean_predictions(prediction):
      prediction = prediction.strip()
      prediction = prediction.rstrip()
      prediction = prediction.strip('<\s>')
      prediction = prediction.strip('</')
      prediction = prediction.strip('</')
      prediction = prediction.strip('[')
      prediction = prediction.strip(']')
      prediction = prediction.strip('.')
      return prediction

  clean_predicted_labels = [clean_predictions(prediction) for prediction in predicted_labels]


  with open(os.path.join(generated_output_path, f'predictions_{split}.txt'), 'w') as file:
     for label in clean_predicted_labels:
        file.write(f"<PREDICTION:{label}>\n")
