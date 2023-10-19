import numpy as np
import os
import pandas as pd
from   pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import time
import torch

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate


formalism_dir = os.getcwd()
labeled_data_path = os.path.join(formalism_dir, 'labeled_data', 'final_cleaned_paragraphs.csv')
output_path = os.path.join(formalism_dir, 'results', 't5-small_binary')

interpretation_df = pd.read_csv(labeled_data_path)
interpretation_df = interpretation_df[interpretation_df['class'].notna()]
interpretation_df["interpretation"] = np.where(interpretation_df["class"].isin(["FORMAL", "GRAND"]), "interpretation", "no interpretation")

model_name = 't5-small'
device = 'cuda'
tokenizer = AutoTokenizer.from_pretrained(model_name) # load in our tokenizer

def preprocess_function(examples, input_text = "paragraph", output_text = "interpretation"):
    model_inputs = tokenizer(examples[input_text], max_length=1024, truncation=True)
    targets = tokenizer(examples[output_text], max_length=5, truncation=True)
    model_inputs["labels"] = targets["input_ids"]
    return model_inputs

def compare_names(df_row, input_text = "paragraph", output_text = "interpretation"):
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


device_name = 'cuda'
macro_f1_l = []
macro_precision_l = []
macro_recall_l = []

weighted_f1_l = []
weighted_precision_l = []
weighted_recall_l = []

one_f1_l = []
one_precision_l = []
one_recall_l = []

zero_f1_l = []
zero_precision_l = []
zero_recall_l = []

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
    result_df = pd.DataFrame(output, columns = ['paragraph', 'interpretation', 'gen_class'])
    print(result_df["gen_class"].tolist())

    class_report = classification_report(result_df["interpretation"].tolist(), result_df["gen_class"].tolist(), output_dict=True)

    sample_dict = {
        "model": "t5_small_finetuned_binary",
        "split": split,

        "macro_f1": round(class_report["macro avg"]["f1-score"], 3),
        "macro_precision": round(class_report["macro avg"]["precision"], 3),
        "macro_recall": round(class_report["macro avg"]["recall"], 3),

        "weighted_f1": round(class_report["weighted avg"]["f1-score"], 3),
        "weighted_precision": round(class_report["weighted avg"]["precision"], 3),
        "weighted_recall": round(class_report["weighted avg"]["recall"], 3),

        "1_f1": round(class_report["interpretation"]["f1-score"], 3),
        "1_precision": round(class_report["interpretation"]["precision"], 3),
        "1_recall": round(class_report["interpretation"]["recall"], 3),

        "0_f1": round(class_report["no interpretation"]["f1-score"], 3),
        "0_precision": round(class_report["no interpretation"]["precision"], 3),
        "0_recall": round(class_report["no interpretation"]["recall"], 3),
    }

    new_row = pd.DataFrame(sample_dict, index = [0])
    full_df = pd.concat([full_df, new_row])

    macro_f1_l.append(class_report["macro avg"]["f1-score"])
    macro_precision_l.append(class_report["macro avg"]["precision"])
    macro_recall_l.append(class_report["macro avg"]["recall"])

    weighted_f1_l.append(class_report["weighted avg"]["f1-score"])
    weighted_precision_l.append(class_report["weighted avg"]["precision"])
    weighted_recall_l.append(class_report["weighted avg"]["recall"])

    one_f1_l.append(class_report["interpretation"]["f1-score"])
    one_precision_l.append(class_report["interpretation"]["precision"])
    one_recall_l.append(class_report["interpretation"]["recall"])

    zero_f1_l.append(class_report["no interpretation"]["f1-score"])
    zero_precision_l.append(class_report["no interpretation"]["precision"])
    zero_recall_l.append(class_report["no interpretation"]["recall"])

macro_f1 = sum(macro_f1_l) / len(macro_f1_l)
macro_precision = sum(macro_precision_l) / len(macro_precision_l)
macro_recall = sum(macro_recall_l) / len(macro_recall_l)

weighted_f1 = sum(weighted_f1_l) / len(weighted_f1_l)
weighted_precision = sum(weighted_precision_l) / len(weighted_precision_l)
weighted_recall = sum(weighted_recall_l) / len(weighted_recall_l)

one_f1 = sum(one_f1_l) / len(one_f1_l)
one_precision = sum(one_precision_l) / len(one_precision_l)
one_recall = sum(one_recall_l) / len(one_recall_l)

zero_f1 = sum(zero_f1_l) / len(zero_f1_l)
zero_precision = sum(zero_precision_l) / len(zero_precision_l)
zero_recall = sum(zero_recall_l) / len(zero_recall_l)


model_dict = {
    "model": "t5_small_finetuned_binary",
    "split": "averages",

    "macro_f1": round(macro_f1, 3),
    "macro_precision": round(macro_precision, 3),
    "macro_recall": round(macro_recall, 3),

    "macro_f1": round(macro_f1, 3),
    "macro_precision": round(macro_precision, 3),
    "macro_recall": round(macro_recall, 3),

    "weighted_f1": round(weighted_f1, 3),
    "weighted_precision": round(weighted_precision, 3),
    "weighted_recall": round(weighted_recall, 3),

    "1_f1": round(one_f1, 3),
    "1_precision": round(one_precision, 3),
    "1_recall": round(one_recall, 3),

    "0_f1": round(zero_f1, 3),
    "0_precision": round(zero_precision, 3),
    "0_recall": round(zero_recall, 3),

}
new_row = pd.DataFrame(model_dict, index = [0])
full_df = pd.concat([full_df, new_row])

full_df.to_csv(os.path.join(output_path, 't5_small_binary_results.csv'))

