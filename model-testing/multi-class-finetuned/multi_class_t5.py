import numpy as np
import os
import pandas as pd
from   pathlib import Path
from sklearn.metrics import  classification_report
import time
import torch

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate


formalism_dir = os.getcwd()
labeled_data_path = os.path.join(formalism_dir, 'labeled_data', 'final_cleaned_paragraphs.csv')
output_path = os.path.join(formalism_dir, 'results', 't5-base')


interpretation_df = pd.read_csv(labeled_data_path)
interpretation_df = interpretation_df[interpretation_df['class'].notna()]

device_name = 'cuda'

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

model_name = 't5-base'
device = 'cuda'
tokenizer = AutoTokenizer.from_pretrained(model_name) 

def preprocess_function(examples, input_text = "paragraph", output_text = "class"):
    model_inputs = tokenizer(examples[input_text], max_length=1024, truncation=True)
    targets = tokenizer(examples[output_text], max_length=5, truncation=True)
    model_inputs["labels"] = targets["input_ids"]
    return model_inputs

def compare_names(df_row, input_text = "paragraph", output_text = "class"):
  inputs = tokenizer(df_row[input_text], return_tensors="pt")
  result = model.generate(inputs=inputs["input_ids"].to(device), max_new_tokens=50)
  gen_class = tokenizer.decode(result[0], skip_special_tokens = True)
  og_class = df_row[output_text]
  output = {
      'text' : df_row[input_text],
      'gen_class' : gen_class,
      'og_class' : og_class,
      }
  return output

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

full_df = pd.DataFrame()

for split in range(0, 5):
    split_id_file = os.path.join(formalism_dir, 'train_test_splits', f'split_{split}')

    with open(split_id_file, 'r') as file:
        train_ids = file.read().split("\n")

    interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
    interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]

    train_dataset = Dataset.from_pandas(interpretation_train_df)
    test_dataset = Dataset.from_pandas(interpretation_test_df)

    tokenized_train_interp = train_dataset.map(preprocess_function, batched = True)
    tokenized_test_interp = test_dataset.map(preprocess_function, batched = True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device=device)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=6,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_interp,
        eval_dataset=tokenized_test_interp,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    output = tokenized_test_interp.map(compare_names)
    result_df = pd.DataFrame(output, columns = ['paragraph', 'class', 'gen_class'])
    print(result_df["gen_class"].tolist())

    class_report = classification_report(result_df["class"].tolist(), result_df["gen_class"].tolist(), output_dict=True)

    sample_dict = {
        "model": "t5_finetuned",
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
    "model": "t5_finetuned",
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

full_df.to_csv(os.path.join(output_path, 't5_base_finetuned_results.csv'))

