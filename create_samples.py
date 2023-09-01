import pandas as pd
import os
import json
import pickle
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from eyecite import get_citations
import argparse

def load_file(file_name):
    """
    Load in the JSONL file. This data has already been filtered by court and year. 
    """
    parsed_data = []
    with open(os.path.join(os.getcwd(), 'data', file_name)) as f:
        for line in f:
            parsed_data.append(line)
    return parsed_data

def clean_text(text):
    """
    Clean strings with multiple numbers and asterisks. 
    """
    text = re.sub("(\*+\d+)", "", text)
    text = re.sub("(\.\d+)", ".", text)
    return text

def find_and_replace_citations(text):
    """
    Use the eyecite package to find citations of court cases. 
    For every citation identified, replace with "CITE".
    """
    try:
        citations = get_citations(text)
        new_text = text
        for citation in citations:
            new_text = re.sub(citation.matched_text(), "CITE", new_text)
    except:
        new_text = text
    return new_text

def check_for_close_substrings(string):
    """
    Check for tokens that occur close (between 200 characters) to each other as a proxy 
    for whether a statute is interpreted. 
    """
    check_1 = re.search(r'(statute|legislation|\sact[\.|\s|,|;]).{0,200}(mean|constru|interpret|reading|understand)', string.lower())
    check_2 = re.search(r'(constru|interpret|reading|understand|mean).{0,200}(statute|legislation|\sact[\.|\s|,|;])', string.lower())
    return any([check_1, check_2])

def convert_to_clean_df(data_dic):
    new_l = []
    for case in data_dic:
        case = json.loads(case)
        case["text"] = case["casebody"]["data"]
        case["pretty_text"] = BeautifulSoup(case['text'], "html.parser").text
        case["pretty_text"] = find_and_replace_citations(case["pretty_text"])
        case["pretty_text"] = clean_text(case["pretty_text"])
        case["word_count"] = len(word_tokenize(case["pretty_text"]))
        new_l.append(case)
    df = pd.DataFrame(new_l)
    longer_cases_df = df[df["word_count"] > 200]
    longer_cases_df["close_stat_interp"] = longer_cases_df.apply(lambda x: check_for_close_substrings(x["pretty_text"]), axis = 1)
    longer_cases_df = longer_cases_df[longer_cases_df['close_stat_interp']]
    return longer_cases_df

def split_to_paragraphs(df):
    """
    Split data into paragraphs. 
    Remove any paragraphs that are less than 200 characters long. 
    """
    paragraph_df = df.assign(split_text=df['pretty_text'].str.split('\n')).explode('split_text')
    paragraph_df = paragraph_df.loc[paragraph_df['split_text'].str.len() > 200]
    paragraph_df = paragraph_df.rename(columns = {"split_text": "paragraph"})
    paragraph_df['sequence']=paragraph_df.groupby('id').cumcount()
    paragraph_df["section_id"] = paragraph_df.apply(lambda x: str(x["id"]) + "_" + str(x["sequence"]), axis = 1)
    paragraph_df["par_word_count"] = paragraph_df.apply(lambda x: len(word_tokenize(x["paragraph"])), axis = 1)
    return paragraph_df

def count_grand_seeds(df): 
    """
    Return words/phrases found in paragraph that are likely grand. 
    """
    p_report_phrases = ["conference report", "committee report", "senate report", "house report", "assembly report"]
    p_talk_words = ["senate hearing", "house hearing", "assembly hearing", "committee hearing", "conference hearing", "floor debate"]
    p_misc_words = ["legislative history", "history of the legislation", "conference committee", "joint committee", "senate committee", "house committee", "assembly committee"]
    p_social_words = ["legislative purpose", "congressional purpose", "purpose of congress", "purpose of the legislature", "social", "society"]
    p_words = p_report_phrases + p_talk_words + p_misc_words + p_social_words
    words_in_paragraph = [term for term in p_words if term in df["paragraph"].lower()]
    return words_in_paragraph 

def count_formal_seeds(df): 
    """
    Return words/phrases found in paragraph that are likely formal. 
    """
    t_dict_words = ["dictionary", "dictionarium", "liguae britannicae", "world book", "funk \& wagnalls"]
    t_grammar_words = ["expressio", "expresio", "inclusio", "noscitur a sociis", "noscitur a socis", "ejusdem generis", "last antecedent", "plain language"]
    t_substantive_words = ["whole act", "whole-act", "whole code", "whole-code", "in pari materia", "meaningful variation", "consistent usage", "surplusage", "superfluit"]
    t_formal_words = ["plain meaning", "ordinary meaning", "word"]
    t_words = t_dict_words + t_grammar_words + t_substantive_words + t_formal_words
    words_in_paragraph = [term for term in t_words if term in df["paragraph"].lower()]
    return words_in_paragraph 

def get_samples(df, size_sample = 100):
    """
    Create samples.
    size_sample determines number of total samples to be created. 
    """
    formal_num = int(.25 * size_sample)
    grand_num = int(.25 * size_sample)
    null_num = int(.50 * size_sample )
    abs_top_num = int(.25 * size_sample) * 4

    df = df.sort_values(by = "diff_percent", ascending = False)
    formal_rows = df.head(abs_top_num).sample(formal_num)
    formal_rows["type"] = "formal"

    grand_rows = df.tail(abs_top_num).sample(grand_num)
    grand_rows["type"] = "grand"

    null_rows = df[df["diff_percent"] == 0].sample(null_num)
    null_rows["type"] = "null"

    frames = [null_rows, formal_rows, grand_rows]
    full_df = pd.concat(frames)
    full_df = full_df.sample(size_sample).reset_index(drop=True)

    return full_df

def add_paragraph_text(df_cell):
    """
    Add "\n["PARAGRAPH]" token to the end of every paragraph. 
    The purpose of this function is purely related to the annotation interface. 
    """
    new_cell = df_cell + "\n[PARAGRAPH]"
    return new_cell

def save_sample_df(df, output_name, size_sample):
    """
    Save the sample as JSON records, with only necessary metadata. 
    """
    sample_df = get_samples(df, size_sample)
    sample_df["paragraph"] = sample_df["paragraph"].apply(add_paragraph_text)
    sample_df = sample_df[["id", "section_id", "url", "name", "paragraph", "diff", "formal_words", "grand_words", "formal_count", "grand_count"]]
    sample_df.to_json(os.path.join(os.getcwd(), 'prodigy', output_name),
            orient="records",
            lines=True)
    return sample_df

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type = str)
    parser.add_argument('--size_sample', type = int)

    print("Parsing arguments.")
    args = parser.parse_args()

    output_name = args.output_name
    size_sample = args.size_sample

    print("Loading data.")
    data = load_file('filtered_data.jsonl')
    print("Converting to clean dataframe.")
    longer_cases_df = convert_to_clean_df(data)
    print("Splitting to paragraphs.")
    paragraph_df = split_to_paragraphs(longer_cases_df)

    print("Checking for formal and grand seeds.")
    paragraph_df["formal_words"] = paragraph_df.apply(count_formal_seeds, axis = 1)
    paragraph_df["grand_words"] = paragraph_df.apply(count_grand_seeds, axis = 1)

    paragraph_df["grand_count"] = paragraph_df.apply(lambda x: len(x["grand_words"]), axis = 1)
    paragraph_df["formal_count"] = paragraph_df.apply(lambda x: len(x["formal_words"]), axis = 1)

    paragraph_df["diff"] = paragraph_df["formal_count"] - paragraph_df["grand_count"] 
    paragraph_df["diff_percent"] = paragraph_df["diff"] / paragraph_df["par_word_count"]
    
    print("Saving sample.")
    sample_df = save_sample_df(paragraph_df, output_name, size_sample)

    sample_df.to_csv(os.path.join(os.getcwd(), 'prodigy', f"{output_name.split('.')[0]}.csv"))

    return sample_df

if __name__ == '__main__':
    main()
