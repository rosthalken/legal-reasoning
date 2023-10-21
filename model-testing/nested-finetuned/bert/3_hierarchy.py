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

model_types = ["legal-bert", 'bert', 'distilbert']
device_name = 'cuda'    
max_length = 512
epoch_num = 3

formalism_dir = os.getcwd()
labeled_data_path = os.path.join(formalism_dir, 'labeled_data', 'final_cleaned_paragraphs.csv')
output_path = os.path.join(formalism_dir, 'results', 'hierarchical')

errors_dir = os.path.join(formalism_dir, 'errors')

if not os.path.exists(errors_dir):
   os.makedirs(errors_dir)

hierarchical_errors_dir = os.path.join(errors_dir, 'hierarchical')
if not os.path.exists(hierarchical_errors_dir):
   os.makedirs(hierarchical_errors_dir)


interpretation_df = pd.read_csv(labeled_data_path)
interpretation_df = interpretation_df[interpretation_df['class'].notna()]
interpretation_df["interpretation"] = np.where(interpretation_df["class"].isin(["FORMAL", "GRAND"]), 1, 0)

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

    for split in range(0, 30):

        interpretation_model_path = os.path.join(output_path, 'models', f'{model_type}_interpretation_{split}_test')
        interpretation_class_model_path = os.path.join(output_path, 'models', f'{model_type}interpretation_class_{split}_test')
        split_id_file = os.path.join(formalism_dir, 'train_test_splits', f'split_{split}')

        with open(split_id_file, 'r') as file:
            train_ids = file.read().split("\n")

        interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
        interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]


        X_train = interpretation_train_df["paragraph"].to_list()
        y_train = interpretation_train_df["class"].to_list()

        X_test = interpretation_test_df["paragraph"].to_list()
        y_test = interpretation_test_df["class"].to_list()



        if model_type == "distilbert":
            model_name = 'distilbert-base-uncased'  
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_name) # The model_name needs to match our pre-trained model.
            interp_model = DistilBertForSequenceClassification.from_pretrained(interpretation_model_path, num_labels=2).to(device_name)
            interp_class_model = DistilBertForSequenceClassification.from_pretrained(interpretation_class_model_path, num_labels=2).to(device_name)
        elif model_type == "legal-bert":
            model_name = "nlpaueb/legal-bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            interp_model = AutoModelForSequenceClassification.from_pretrained(interpretation_model_path, num_labels=2).to(device_name)
            interp_class_model = AutoModelForSequenceClassification.from_pretrained(interpretation_class_model_path, num_labels=2).to(device_name)
        elif model_type == "bert":
            model_name = "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            interp_model = AutoModelForSequenceClassification.from_pretrained(interpretation_model_path, num_labels=2).to(device_name)
            interp_class_model = AutoModelForSequenceClassification.from_pretrained(interpretation_class_model_path, num_labels=2).to(device_name)

        else:
            print("No model with this name.")

        batchsize = 8
        interpretation_predictions = []
        worklist = X_test

        for i in range(0, len(worklist), batchsize):
            batch = worklist[i:i+batchsize] # extract batch from worklist
            test_encodings = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device_name)
            output = interp_model(**test_encodings) # make predictions with model on our test_encodings for this batch
            batch_predictions = torch.softmax(output.logits, dim=1).tolist() # get the predictions result
            interpretation_predictions.append(batch_predictions)

        flat_list = [item for sublist in interpretation_predictions for item in sublist]

        interpretation_predictions_list = [list_items.index(max(list_items)) for list_items in flat_list]


        class_predictions = []

        for prediction, text in zip(interpretation_predictions_list, X_test):
            if prediction == 1:
                test_encodings = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device_name)
                output = interp_class_model(**test_encodings) # make predictions with model on our test_encodings for this batch
                predicted_probs = torch.softmax(output.logits, dim=1).tolist()[0] # get the predictions result
                probable_idx = predicted_probs.index(max(predicted_probs))
                class_predictions.append(probable_idx)
            else:
                class_predictions.append(2)


        name_dic = {
            Counter(class_predictions).most_common()[2][0]:"FORMAL",
            Counter(class_predictions).most_common()[1][0]: "GRAND",
            Counter(class_predictions).most_common()[0][0]: "NONE"
        }

        interpretation_test_df["predicted_nums"] = class_predictions
        interpretation_test_df["predicted_classes"] = interpretation_test_df["predicted_nums"].map(name_dic)
        y_predicted = interpretation_test_df["predicted_classes"].tolist()


        predictions_df = pd.DataFrame(
            {'section_id': interpretation_test_df["section_id"].tolist(),
            'gold': y_test,
            'predicted': y_predicted,
            'text': X_test
            })
        errors_df = predictions_df.query('gold != predicted')
        errors_df.to_csv(os.path.join(hierarchical_errors_dir, f"{model_type}_{split}_errors.csv"))


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