import pandas as pd
import numpy as np
import os
from collections import Counter


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from datasets import load_dataset, Dataset
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments


model_types = ["t5-base", "t5-small"]
device_name = 'cuda'    
max_length = 512
epoch_num = 3


formalism_dir = os.getcwd()
labeled_data_path = os.path.join(formalism_dir, 'labeled_data', 'final_cleaned_paragraphs.csv')
output_path = os.path.join(formalism_dir, 'results', 't5-hierarchical')


interpretation_df = pd.read_csv(labeled_data_path)
interpretation_df = interpretation_df[interpretation_df['class'].notna()]
interpretation_df["interpretation"] = np.where(interpretation_df["class"].isin(["FORMAL", "GRAND"]), "interpretation", "no interpretation")

def preprocess_function(examples, input_text = "paragraph", output_text = "class"):
    model_inputs = tokenizer(examples[input_text], max_length=1024, truncation=True)
    targets = tokenizer(examples[output_text], max_length=5, truncation=True)
    model_inputs["labels"] = targets["input_ids"]
    return model_inputs


for model_type in model_types:
    for split in range(0, 5):
        interpretation_class_model_path = os.path.join(output_path, 'models', f'{model_type}interpretation_class_{split}_test')
        split_id_file = os.path.join(formalism_dir, 'train_test_splits', f'split_{split}')

        with open(split_id_file, 'r') as file:
            train_ids = file.read().split("\n")

        interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
        interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]

        interpretation_train_df = interpretation_train_df[interpretation_df['class'] != "NONE"] 
        interpretation_test_df = interpretation_test_df[interpretation_test_df['class'] != "NONE"]

        tokenizer = AutoTokenizer.from_pretrained(model_type)

        train_dataset = Dataset.from_pandas(interpretation_train_df)
        test_dataset = Dataset.from_pandas(interpretation_test_df)

        tokenized_train_interp = train_dataset.map(preprocess_function, batched = True)
        tokenized_test_interp = test_dataset.map(preprocess_function, batched = True)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_type).to(device=device_name)
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

        model.save_pretrained(interpretation_class_model_path)

        trainer.evaluate()