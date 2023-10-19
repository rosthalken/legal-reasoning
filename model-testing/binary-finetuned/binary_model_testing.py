import pandas as pd
import numpy as np
import os
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import confusion_matrix #, ConfusionMatrixDisplay


# import evaluate
from datasets import load_dataset, Dataset
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments



device_name = 'cuda'    
max_length = 512

formalism_dir = os.getcwd()
labeled_data_path = os.path.join(formalism_dir, 'labeled_data', 'final_cleaned_paragraphs.csv')
output_path = os.path.join(formalism_dir, 'results')
fig_path = os.path.join(formalism_dir, 'figures')

labels = ["interpretation", "formal", "grand", "none"]
models = ["bert", "legal-bert", "distilbert"] 
epoch_num = 3

interpretation_df = pd.read_csv(labeled_data_path)
interpretation_df = interpretation_df[interpretation_df['class'].notna()]

interpretation_df["formal"] = np.where(interpretation_df["class"] == "FORMAL", 1, 0)
interpretation_df["grand"] = np.where(interpretation_df["class"] == "GRAND", 1, 0)
interpretation_df["interpretation"] = np.where(interpretation_df["class"].isin(["FORMAL", "GRAND"]), 1, 0)

final_df = pd.DataFrame()

for label_name in labels:
  full_df = pd.DataFrame()
  
  for model_type in models:
  
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

      
      for split in range(0, 10):

        split_id_file = os.path.join(formalism_dir, 'train_test_splits', f'split_{split}')

        with open(split_id_file, 'r') as file:
            train_ids = file.read().split("\n")

        interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
        interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]


        X_train = interpretation_train_df["paragraph"].to_list()
        y_train = interpretation_train_df[label_name].to_list()

        X_test = interpretation_test_df["paragraph"].to_list()
        y_test = interpretation_test_df[label_name].to_list()


        unique_labels = set(label for label in y_train)
        label2id = {label: id for id, label in enumerate(unique_labels)}
        id2label = {id: label for label, id in label2id.items()}


        if model_type == "distilbert":
          model_name = 'distilbert-base-uncased'  
          tokenizer = DistilBertTokenizerFast.from_pretrained(model_name) # The model_name needs to match our pre-trained model.
          model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(id2label)).to(device_name)
        elif model_type == "legal-bert":
          model_name = "nlpaueb/legal-bert-base-uncased"
          tokenizer = AutoTokenizer.from_pretrained(model_name)
          model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(id2label)).to(device_name)
        elif model_type == "bert":
          model_name = "bert-base-uncased"
          tokenizer = AutoTokenizer.from_pretrained(model_name)
          model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(id2label)).to(device_name)
        else:
          print("No model with this name.")
        
        train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=max_length)
        test_encodings  = tokenizer(X_test, truncation=True, padding=True, max_length=max_length)

        train_labels_encoded = [label2id[y] for y in y_train]
        test_labels_encoded  = [label2id[y] for y in y_test]


        class MyDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset = MyDataset(train_encodings, train_labels_encoded)
        test_dataset = MyDataset(test_encodings, test_labels_encoded)

        training_args = TrainingArguments(
            num_train_epochs=epoch_num, # total number of training epochs
            optim="adamw_torch",
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=20,   # batch size for evaluation
            learning_rate=2e-5,              # initial learning rate for Adam optimizer
            warmup_steps=50,                 # number of warmup steps for learning rate scheduler (set lower because of small dataset size)
            weight_decay=0.01,               # strength of weight decay
            output_dir='./results',          # output directory
            logging_dir='training_logs',     # directory for storing logs
            logging_steps=20,                # number of steps to output logging (set lower because of small dataset size)
            evaluation_strategy='steps',     # evaluate during fine-tuning so that we can see progress
        )

        def compute_metrics(pred):
          labels = pred.label_ids
          preds = pred.predictions.argmax(-1)
          acc = accuracy_score(labels, preds)
          return {
              'accuracy': acc,
          }

        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=test_dataset,           # evaluation dataset (usually a validation set; here we just send our test set)
            compute_metrics=compute_metrics      # our custom evaluation function 
        )

        trainer.train()  



        trainer.evaluate()

        predicted_results = trainer.predict(test_dataset)

        predicted_labels = predicted_results.predictions.argmax(-1) # Get the highest probability prediction
        predicted_labels = predicted_labels.flatten().tolist()      # Flatten the predictions into a 1D list
        predicted_labels = [id2label[l] for l in predicted_labels]  # Convert from integers back to strings for readability

        class_report = classification_report(y_test, predicted_labels, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()

        macro_f1_l.append(class_report["macro avg"]["f1-score"])
        macro_precision_l.append(class_report["macro avg"]["precision"])
        macro_recall_l.append(class_report["macro avg"]["recall"])

        weighted_f1_l.append(class_report["weighted avg"]["f1-score"])
        weighted_precision_l.append(class_report["weighted avg"]["precision"])
        weighted_recall_l.append(class_report["weighted avg"]["recall"])

        one_f1_l.append(class_report["1"]["f1-score"])
        one_precision_l.append(class_report["1"]["precision"])
        one_recall_l.append(class_report["1"]["recall"])

        zero_f1_l.append(class_report["0"]["f1-score"])
        zero_precision_l.append(class_report["0"]["precision"])
        zero_recall_l.append(class_report["0"]["recall"])


        sample_dict = {
            "label": label_name,
            "model": model_type,
            "split": split,

            "macro_f1": round(class_report["macro avg"]["f1-score"], 3),
            "macro_precision": round(class_report["macro avg"]["precision"], 3), 
            "macro_recall": round(class_report["macro avg"]["recall"], 3), 

            "weighted_f1": round(class_report["weighted avg"]["f1-score"], 3),
            "weighted_precision": round(class_report["weighted avg"]["precision"], 3), 
            "weighted_recall": round(class_report["weighted avg"]["recall"], 3), 

            "1_f1": round(class_report["1"]["f1-score"], 3),
            "1_precision": round(class_report["1"]["precision"], 3), 
            "1_recall": round(class_report["1"]["recall"], 3), 

            "0_f1": round(class_report["0"]["f1-score"], 3),
            "0_precision": round(class_report["0"]["precision"], 3), 
            "0_recall": round(class_report["0"]["recall"], 3), 

            # "none_f1": round(class_report["NONE"]["f1-score"], 3),
            # "none_precision": round(class_report["NONE"]["precision"], 3), 
            # "none_recall": round(class_report["NONE"]["recall"], 3), 
        }

        new_row = pd.DataFrame(sample_dict, index = [0])

        full_df = pd.concat([full_df, new_row])

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
    
      # grand_f1 = sum(grand_f1_l) / len(grand_f1_l)
      # grand_precision = sum(grand_precision_l) / len(grand_precision_l) 
      # grand_recall = sum(grand_recall_l) / len(grand_recall_l)
    
      # formal_f1 = sum(formal_f1_l) / len(formal_f1_l)
      # formal_precision = sum(formal_precision_l) / len(formal_precision_l) 
      # formal_recall = sum(formal_recall_l) / len(formal_recall_l)
    
      # none_f1 = sum(none_f1_l) / len(none_f1_l)
      # none_precision = sum(none_precision_l) / len(none_precision_l) 
      # none_recall = sum(none_recall_l) / len(none_recall_l)




      # create all the things to save
      sample_dict = {
          "label": label_name,
          "model": model_type,
          "split": "averages",

          "macro_f1": round(macro_f1, 3),
          "macro_precision": round(macro_precision, 3), 
          "macro_recall": round(macro_recall, 3), 

          "weighted_f1": round(weighted_f1, 3),
          "weighted_precision": round(weighted_precision, 3), 
          "weighted_recall": round(weighted_recall, 3), 

          "one_f1": round(one_f1, 3),
          "one_precision": round(one_precision, 3), 
          "one_recall": round(one_recall, 3), 

          "zero_f1": round(zero_f1, 3),
          "zero_precision": round(zero_precision, 3), 
          "zero_recall": round(zero_recall, 3), 
      }

      new_row = pd.DataFrame(sample_dict, index = [0])

      full_df = pd.concat([full_df, new_row])

  full_df.to_csv(os.path.join(output_path, f'{label_name}_binary_model_comparison.csv'))
  final_df = pd.concat([final_df, full_df])

final_df.to_csv(os.path.join(output_path, 'binary_model_comparison.csv'))

