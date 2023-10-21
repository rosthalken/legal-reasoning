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
output_path = os.path.join(formalism_dir, 'results', 'llama_context')
generated_output_path = os.path.join(output_path, 'generations')
descriptive_errors_dir = os.path.join(output_path, 'errors')

device_name = 'cuda'

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(device_name)

prompt_text = "Some paragraphs in court cases interpret statutes. Within interpretation, there are two types:  grand and formal. \n\nGrand interpretation represents a legal decision that views law as an open-ended and on-going enterprise for the production and improvement of decisions that make sense on their face and in light of political, social, and economic factors.  \n\nFormal interpretation is a legal decision made according to a rule, often viewing the law as a closed and mechanical system. It screens the decision-maker off from the political, social, and economic choices involved in the decision.  \n\nLet's analyze the following passage step-by-step. First, determine if it interprets a statute. Second, if it interprets a statute, determine whether the interpretation is grand or formal. The first word in your response should label the passage with \"GRAND\", \"FORMAL\", or \"NONE\" and then explain why you chose that label. \n\n"

inst_prompt = f"""
<s>[INST] <<SYS>>
{prompt_text}
<</SYS>>
"""

def get_interp_type(inst_prompt, text):
    full_prompt = f"{inst_prompt}\n\"{text}\"\n\nYou must respond in a single word. Your options are either \"GRAND\", \"FORMAL\", or \"NONE\". What is the one word that describes this paragraph? [/INST]"
    print(full_prompt)
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

  with open(os.path.join(generated_output_path, f'predictions_{split}.txt'), 'w') as file:
     for label in predicted_labels:
        file.write(f"<PREDICTION:{label}>\n")




