import pandas as pd
import numpy as np
import os

from sklearn.metrics import accuracy_score, classification_report

from datasets import load_dataset, Dataset
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

device_name = 'cuda'    
max_length = 512

formalism_dir = os.getcwd()
labeled_data_path = os.path.join(formalism_dir, 'labeled_data', 'final_cleaned_paragraphs.csv')
output_path = os.path.join(formalism_dir, 'results',  'multi-class')

model_name = "nlpaueb/legal-bert-base-uncased"
epoch_num = 3

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


split_id_file = os.path.join(formalism_dir, 'train_test_splits', f'split_0')
with open(split_id_file, 'r') as file:
    train_ids = file.read().split("\n")

interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]


X_train = interpretation_train_df["paragraph"].to_list()
y_train = interpretation_train_df["class"].to_list()

X_test = interpretation_test_df["paragraph"].to_list()
y_test = interpretation_test_df["class"].to_list()

unique_labels = set(label for label in y_train)
label2id = {label: id for id, label in enumerate(unique_labels)}
id2label = {id: label for label, id in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(id2label)).to(device_name)

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
    num_train_epochs=epoch_num, 
    optim="adamw_torch",
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=20, 
    learning_rate=2e-5,             
    warmup_steps=50,              
    weight_decay=0.01,             
    output_dir='./results',       
    logging_dir='training_logs',   
    logging_steps=20,              
    evaluation_strategy='steps',   
)

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

trainer = Trainer(
    model=model,                        
    args=training_args,              
    train_dataset=train_dataset,      
    eval_dataset=test_dataset,      
    compute_metrics=compute_metrics   
)

trainer.train()  
trainer.evaluate()

predicted_results = trainer.predict(test_dataset)
predicted_labels = predicted_results.predictions.argmax(-1) 
predicted_labels = predicted_labels.flatten().tolist() 
predicted_labels = [id2label[l] for l in predicted_labels]  

predictions_df = pd.DataFrame(
{'section_id': interpretation_test_df["section_id"].tolist(),
  'gold': y_test,
  'predicted': predicted_labels,
  'text': X_test
})

class_report = classification_report(y_test, predicted_labels, output_dict=True)
classification_df = pd.DataFrame(class_report)

classification_df.to_csv(os.path.join(output_path, f'trained_model.csv'))


