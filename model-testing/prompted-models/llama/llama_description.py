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
output_path = os.path.join(formalism_dir, 'results', 'llama_descriptive')
generated_output_path = os.path.join(output_path, 'generations')
descriptive_errors_dir = os.path.join(output_path, 'errors')

device_name = 'cuda'

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device_name)

intro_statement = "There are three possible labels to describe legal interpretation in the following passage: FORMAL, GRAND, or NONE."
formal_description = "FORMAL theory is a legal decision made according to a rule, often viewing the law as a closed and mechanical system. It screens the decision-maker off from the political, social, and economic choices involved in the decision. "
grand_description = "GRAND theory is legal decision that views law as an open-ended and on-going enterprise for the production and improvement of decisions that make sense on their face and in light of political, social, and economic factors."
none_description = "NONE is a passage or mode of reasoning that does not reflect either the Grand or Formal approaches. Note that this coding would include areas of substantive law outside of statutory interpretation, including procedural matters. "

prompt = f"{intro_statement} \n\n {formal_description}\n\n {grand_description}\n\n {none_description}\n\n"

def get_interp_type(prompt, text):
    full_prompt = f"{prompt} \n\n Text: {text} \n\nChoice of GRAND, FORMAL, or NONE: \n\n"
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device_name)
    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=2
    )
    output_text = tokenizer.batch_decode(generation_output[:, input_ids.shape[1]:])[0]
    return output_text





full_df = pd.DataFrame()
interpretation_df = pd.read_csv(labeled_data_path)
interpretation_df = interpretation_df[interpretation_df['class'].notna()]

macro_f1_l = []
macro_precision_l = []
macro_recall_l = []

weighted_f1_l = []
weighted_precision_l = []
weighted_recall_l = []

grand_f1_l = []
grand_precision_l = []
grand_recall_l = []

formal_f1_l = []
formal_precision_l = []
formal_recall_l = []

none_f1_l = []
none_precision_l = []
none_recall_l = []


for split in range(0, 1):

  split_id_file = os.path.join(formalism_dir, 'train_test_splits', f'split_{split}')

  with open(split_id_file, 'r') as file:
      train_ids = file.read().split("\n")

  interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
  interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]


  X_test = interpretation_test_df["paragraph"].to_list()
  y_test = interpretation_test_df["class"].to_list()
  
  predicted_labels = [get_interp_type(prompt, text).upper() for text in X_test]

  with open(os.path.join(generated_output_path, f'predictions_{split}.txt'), 'w') as file:
     for label in predicted_labels:
        file.write(f"{label}\n")


  predictions_df = pd.DataFrame(
  {'section_id': interpretation_test_df["section_id"].tolist(),
    'gold': y_test,
    'predicted': predicted_labels,
    'text': X_test
  })
  errors_df = predictions_df.query('gold != predicted')
  errors_df.to_csv(os.path.join(descriptive_errors_dir, f"{split}_errors.csv"))

  class_report = classification_report(y_test, predicted_labels, output_dict=True)

  sample_dict = {
      "model": "descriptive_generative",
      "split": split,

      "macro_f1": round(class_report["macro avg"]["f1-score"], 3),
      "macro_precision": round(class_report["macro avg"]["precision"], 3), 
      "macro_recall": round(class_report["macro avg"]["recall"], 3), 

      "weighted_f1": round(class_report["weighted avg"]["f1-score"], 3),
      "weighted_precision": round(class_report["weighted avg"]["precision"], 3), 
      "weighted_recall": round(class_report["weighted avg"]["recall"], 3), 

      "grand_f1": round(class_report["GRAND"]["f1-score"], 3),
      "grand_precision": round(class_report["GRAND"]["precision"], 3), 
      "grand_recall": round(class_report["GRAND"]["recall"], 3), 

      "formal_f1": round(class_report["FORMAL"]["f1-score"], 3),
      "formal_precision": round(class_report["FORMAL"]["precision"], 3), 
      "formal_recall": round(class_report["FORMAL"]["recall"], 3), 

      "none_f1": round(class_report["NONE"]["f1-score"], 3),
      "none_precision": round(class_report["NONE"]["precision"], 3), 
      "none_recall": round(class_report["NONE"]["recall"], 3), 
  }

  new_row = pd.DataFrame(sample_dict, index = [0])
  full_df = pd.concat([full_df, new_row])


  macro_f1_l.append(class_report["macro avg"]["f1-score"])
  macro_precision_l.append(class_report["macro avg"]["precision"])
  macro_recall_l.append(class_report["macro avg"]["recall"])

  weighted_f1_l.append(class_report["weighted avg"]["f1-score"])
  weighted_precision_l.append(class_report["weighted avg"]["precision"])
  weighted_recall_l.append(class_report["weighted avg"]["recall"])

  grand_f1_l.append(class_report["GRAND"]["f1-score"])
  grand_precision_l.append(class_report["GRAND"]["precision"])
  grand_recall_l.append(class_report["GRAND"]["recall"])

  formal_f1_l.append(class_report["FORMAL"]["f1-score"])
  formal_precision_l.append(class_report["FORMAL"]["precision"])
  formal_recall_l.append(class_report["FORMAL"]["recall"])

  none_f1_l.append(class_report["NONE"]["f1-score"])
  none_precision_l.append(class_report["NONE"]["precision"])
  none_recall_l.append(class_report["NONE"]["recall"])


macro_f1 = sum(macro_f1_l) / len(macro_f1_l)
macro_precision = sum(macro_precision_l) / len(macro_precision_l) 
macro_recall = sum(macro_recall_l) / len(macro_recall_l)

weighted_f1 = sum(weighted_f1_l) / len(weighted_f1_l)
weighted_precision = sum(weighted_precision_l) / len(weighted_precision_l) 
weighted_recall = sum(weighted_recall_l) / len(weighted_recall_l)

grand_f1 = sum(grand_f1_l) / len(grand_f1_l)
grand_precision = sum(grand_precision_l) / len(grand_precision_l) 
grand_recall = sum(grand_recall_l) / len(grand_recall_l)

formal_f1 = sum(formal_f1_l) / len(formal_f1_l)
formal_precision = sum(formal_precision_l) / len(formal_precision_l) 
formal_recall = sum(formal_recall_l) / len(formal_recall_l)

none_f1 = sum(none_f1_l) / len(none_f1_l)
none_precision = sum(none_precision_l) / len(none_precision_l) 
none_recall = sum(none_recall_l) / len(none_recall_l)


model_dict = {
    "model": "descriptive_generative",
    "split": "averages",

    "macro_f1": round(macro_f1, 3),
    "macro_precision": round(macro_precision, 3), 
    "macro_recall": round(macro_recall, 3), 

    "weighted_f1": round(weighted_f1, 3),
    "weighted_precision": round(weighted_precision, 3), 
    "weighted_recall": round(weighted_recall, 3), 

    "grand_f1": round(grand_f1, 3),
    "grand_precision": round(grand_precision, 3), 
    "grand_recall": round(grand_recall, 3), 

    "formal_f1": round(formal_f1, 3),
    "formal_precision": round(formal_precision, 3), 
    "formal_recall": round(formal_recall, 3), 

    "none_f1": round(none_f1, 3),
    "none_precision": round(none_precision, 3), 
    "none_recall": round(none_recall, 3), 
}

new_row = pd.DataFrame(model_dict, index = [0])
full_df = pd.concat([full_df, new_row])

full_df.to_csv(os.path.join(output_path, 'llama_generated_results.csv'))


