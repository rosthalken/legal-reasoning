import pandas as pd
import numpy as np
import os
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


# import evaluate
from datasets import load_dataset, Dataset
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments


model_types = ["legal-bert", 'bert', 'distilbert']
device_name = 'cuda'    
max_length = 512
epoch_num = 3


formalism_dir = os.getcwd()
labeled_data_path = os.path.join(formalism_dir, 'labeled_data', 'final_cleaned_paragraphs.csv')
output_path = os.path.join(formalism_dir, 'results', 'hierarchical')

interpretation_df = pd.read_csv(labeled_data_path)
interpretation_df = interpretation_df[interpretation_df['class'].notna()]
interpretation_df["interpretation"] = np.where(interpretation_df["class"].isin(["FORMAL", "GRAND"]), 1, 0)

for model_type in model_types:
    for split in range(0, 30):
        interpretation_model_path = os.path.join(output_path, 'models', f'{model_type}_interpretation_{split}_test')
        split_id_file = os.path.join(formalism_dir, 'train_test_splits', f'split_{split}')

        with open(split_id_file, 'r') as file:
            train_ids = file.read().split("\n")

        interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
        interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]

        X_train = interpretation_train_df["paragraph"].to_list()
        y_train = interpretation_train_df["interpretation"].to_list()

        X_test = interpretation_test_df["paragraph"].to_list()
        y_test = interpretation_test_df["interpretation"].to_list()

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

        model.save_pretrained(interpretation_model_path)

        trainer.evaluate()