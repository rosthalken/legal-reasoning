import os
import time
from collections import defaultdict
import random
import pickle
import pandas as pd
import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

start_time = time.time()
device_name = 'cuda'       
max_length = 512      

formalism_dir = os.getcwd()
dataset_path = os.path.join(formalism_dir, 'data', 'data_for_predictions.csv')
model_path = os.path.join(formalism_dir, 'models', 'finetuned-legal-bert')

output_path = os.path.join(formalism_dir,  'predictions')
raw_output_path = os.path.join(output_path, 'raw_output')
combined_output_path = os.path.join(output_path, 'combined_output')
combined_csv_path = os.path.join(output_path, 'combined_output.csv')

model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3).to(device_name)

interp_df = pd.read_csv(dataset_path)
worklist = interp_df["paragraph"].to_list()

batchsize = 8
predictions = []

for i in range(0, len(worklist), batchsize):
    batch = worklist[i:i+batchsize] 
    test_encodings = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device_name)
    output = model(**test_encodings)
    batch_predictions = torch.softmax(output.logits, dim=1).tolist() 
    predictions.append(batch_predictions)
    if i % 1000 == 0:
        print(str(i)+" in "+str(len(worklist)))

pickle.dump(predictions, open(raw_output_path, "wb"))

flat_list = [item for sublist in predictions for item in sublist]
interp_df["predictions"] = flat_list
interp_df[['prob_0','prob_1', 'prob_2']] = pd.DataFrame(interp_df["predictions"].tolist(), index=interp_df.index)
interp_df["predicted_class"] = [list_items.index(max(list_items)) for list_items in flat_list]

interp_df.to_pickle(combined_output_path)
interp_df.to_csv(combined_csv_path)

end_time = time.time()
print(f"PREDICTIONS FINISHED PROCESSING IN {end_time - start_time} SECONDS.")

