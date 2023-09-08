import pandas as pd
import os
import json
import re

file_path = os.path.join(os.getcwd(), 'data', 'data.jsonl')
output_path = os.path.join(os.getcwd(), 'data', 'filtered_data.jsonl')

with open(file_path) as f:
    for line in f:
        item = json.loads(line)
        date = item["decision_date"]
        year = int(re.findall('\d{4,}',date)[0])
        court = item["court"]["name"]
        court_check = re.search(r'supreme court', court.lower()) # updated 
        year_check = year > 1869
        output_file = open(output_path, 'a', encoding='utf-8')
        if court_check and year_check:
                json.dump(item, output_file) 
                output_file.write("\n")
