import numpy as np
import os
import pandas as pd
from   pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import time
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

formalism_dir = os.getcwd()
labeled_data_path = os.path.join(formalism_dir, 'labeled_data', 'final_cleaned_paragraphs.csv')
output_path = os.path.join(formalism_dir, 'results', 'llama_few_shot')
generated_output_path = os.path.join(output_path, 'generations')
descriptive_errors_dir = os.path.join(output_path, 'errors')

device_name = 'cuda'

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(device_name)

prompt_text = "Determine the legal interpretation used in the following passage. Return a single choice from FROMAL, GRAND, or NONE. Here are examples:"
formal_example = "Accepting this point, too, for argument's sake, the question becomes: What did \"discriminate\" mean in 1964? As it turns out, it meant then roughly what it means today: \"To make a difference in treatment or favor (of one as compared with others).\" Webster's New International Dictionary 745 (2d ed. 1954). To \"discriminate against\" a person, then, would seem to mean treating that individual worse than others who are similarly situated. [CITE]. In so-called \"disparate treatment\" cases like today's, this Court has also held that the difference in treatment based on sex must be intentional. See, e.g., [CITE]. So, taken together, an employer who intentionally treats a person worse because of sex—such as by firing the person for actions or attributes it would tolerate in an individual of another sex—discriminates against that person in violation of Title VII."
grand_example = "Respondent's argument is not without force. But it overlooks the significance of the fact that the Kaiser-USWA plan is an affirmative action plan voluntarily adopted by private parties to eliminate traditional patterns of racial segregation. In this context respondent's reliance upon a literal construction of §§ 703 (a) and (d) and upon McDonald is misplaced. See [CITE]. It is a \"familiar rule, that a thing may be within the letter of the statute and yet not within the statute, because not within its spirit, nor within the intention of its makers.\" [CITE]. The prohibition against racial discrimination in §§ 703 (a) and (d) of Title VII must therefore be read against the background of the legislative history of Title VII and the historical context from which the Act arose. See [CITE]. Examination of those sources makes clear that an interpretation of the sections that forbade all race-conscious affirmative action would \"bring about an end completely at variance with the purpose of the statute\" and must be rejected. [CITE]. See [CITE]."
none_example = "The questions are, What is the form of an assignment, and how must it be evidenced? There is no precise form. It may be. by delivery. Briggs v. Dorr, CITE, citing numerous cases; Onion v. Paul, 1 Har. & Johns. 114; Dunn v. Snell, CITE; Titcomb v. Thomas, 5 Greenl. 282. True, it is said it must be on a valuable consideration, with intent to transfer it. But these last are requisites in all assignments, or transfers of securities, negotiable or not. It may be by writing under seal, by writing without seal, by oral declarations, accompanied in all cases by delivery, and on a just consideration. The evidence may be by proof of handwriting and proof of. possession. It may be proved by proving the signature of the payee or obligee on the back, and possession by a third person. 3 Gill & John"

prompt_text = f"{prompt_text}\n\n###\nText: {formal_example}\nFORMAL\n###\nText: {grand_example}\nGRAND\n###\nText: {none_example}\nNONE\n###"

inst_prompt = f"""
<s>[INST] <<SYS>>
{prompt_text}
<</SYS>>
"""

def get_interp_type(inst_prompt, text):
    full_prompt = f"{inst_prompt}\n\"{text}\"\n\nYou must respond in a single word. Your options are any of \"GRAND\", \"FORMAL\", or \"NONE\". What is the one word that describes this paragraph? [/INST]"
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device_name)
    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=10,
    )
    output_text = tokenizer.batch_decode(generation_output[:, input_ids.shape[1]:])[0]
    return output_text

full_df = pd.DataFrame()
interpretation_df = pd.read_csv(labeled_data_path)
interpretation_df = interpretation_df[interpretation_df['class'].notna()]

for split in range(0, 5):

  split_id_file = os.path.join(formalism_dir, 'train_test_splits', f'split_{split}')

  with open(split_id_file, 'r') as file:
      train_ids = file.read().split("\n")

  interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
  interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]


  X_test = interpretation_test_df["paragraph"].to_list()
  y_test = interpretation_test_df["class"].to_list()
  
  predicted_labels = [get_interp_type(inst_prompt, text).upper() for text in X_test]

  with open(os.path.join(generated_output_path, f'predictions_{split}.txt'), 'w') as file:
     for label in predicted_labels:
        file.write(f"<PREDICTION:{label}>\n")



