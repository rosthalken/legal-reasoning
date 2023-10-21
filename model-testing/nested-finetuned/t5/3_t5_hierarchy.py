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

for model_type in model_types:
    full_df = pd.DataFrame()

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

    for split in range(0, 5):

        interpretation_model_path = os.path.join(output_path, 'models', f'{model_type}_interpretation_{split}_test')
        interpretation_class_model_path = os.path.join(output_path, 'models', f'{model_type}interpretation_class_{split}_test')
        split_id_file = os.path.join(formalism_dir, 'train_test_splits', f'split_{split}')

        with open(split_id_file, 'r') as file:
            train_ids = file.read().split("\n")

        interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
        interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]

        train_dataset = Dataset.from_pandas(interpretation_train_df)
        test_dataset = Dataset.from_pandas(interpretation_test_df)

        tokenizer = AutoTokenizer.from_pretrained(model_type) # load in our tokenizer
        interp_model = AutoModelForSeq2SeqLM.from_pretrained(interpretation_model_path).to(device_name)
        interp_class_model = AutoModelForSeq2SeqLM.from_pretrained(interpretation_class_model_path).to(device_name)

        def predict_interp(df_row, input_text = "paragraph", output_text = "class"):
            inputs = tokenizer(df_row[input_text], return_tensors="pt")
            result = interp_model.generate(inputs=inputs["input_ids"].to(device_name), max_new_tokens=5)
            gen_class = tokenizer.decode(result[0], skip_special_tokens = True)
            og_class = df_row[output_text]
            output = {
                'text' : df_row[input_text],
                'generated_interp' : gen_class,
                'class' : og_class,
                }
            return output

        interp_output = test_dataset.map(predict_interp)
        interp_df = pd.DataFrame(interp_output, columns = ['paragraph', 'generated_interp', 'class'])

        interpretation_predictions = interp_df["generated_interp"].tolist()
        paragraphs = interp_df["paragraph"].tolist()

        class_predictions = []

        for prediction, text in zip(interpretation_predictions, paragraphs):
            if prediction == "interpretation":
                inputs = tokenizer(text, return_tensors="pt")
                result = interp_class_model.generate(inputs=inputs["input_ids"].to(device_name), max_new_tokens=5)
                gen_class = tokenizer.decode(result[0], skip_special_tokens = True)
                class_predictions.append(gen_class)
            else:
                class_predictions.append("NONE")

        interp_df["predicted_class"] = class_predictions
        y_predicted = interp_df["predicted_class"].tolist()
        y_test = interp_df["class"].tolist()

        class_report = classification_report(y_test, y_predicted, output_dict=True)

        sample_dict = {
            "model": model_type,
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


    final_dict = {
        "model": model_type,
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

    new_row = pd.DataFrame(final_dict, index = [0])
    full_df = pd.concat([full_df, new_row])
    full_df.to_csv(os.path.join(output_path, f'{model_type}_hierarchical_model_comparison.csv'))