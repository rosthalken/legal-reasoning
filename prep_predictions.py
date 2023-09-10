import pandas as pd
import os
import json
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from eyecite import get_citations

def load_file(file_name):
    """
    Read in pre-filtered data file.
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

def convert_to_clean_df(data_dic): # new function
    """
    Preprocess the text, including finding opinion-level data and replacing citations with "CITE".
    """
    new_l = []
    for case in data_dic:
        case = json.loads(case)
        # collect necessary case metadata
        id = case["id"]
        url = case["url"]
        case_name = case["name"]
        date = case["decision_date"]
        court_url = case["court"]["url"]

        soup = BeautifulSoup(case["casebody"]["data"], "html.parser")
        opinions = soup.find_all("opinion") # find *ALL*
        print(f"opinion number: {len(opinions)}")
        for opinion in opinions:
            # collect necessary opinion metadata
            author_names = []
            author_ids = []
            opinion_type = opinion["type"]
            opinion_text = opinion.text
            pretty_text = find_and_replace_citations(opinion_text)
            pretty_text = clean_text(pretty_text)
            word_count = len(word_tokenize(pretty_text))
            close_stat_interp = check_for_close_substrings(pretty_text)


            authors = opinion.find_all("author")
            print(f"author number: {len(authors)}")
            for author in authors:
                name = author.text
                id = author["id"]
                author_names.append(name)
                author_ids.append(id)
            
            opinion_dict = {
                "id":id,
                "url":url,
                "case_name":case_name,
                "date":date,
                "court_url":court_url,
                "opinion_type":opinion_type,
                "opinion_text":opinion_text,
                "pretty_text":pretty_text,
                "word_count":word_count,
                "close_stat_interp":close_stat_interp,
                "author_names":author_names,
                "author_ids":author_ids
            }
            new_l.append(opinion_dict)
    df = pd.DataFrame(new_l)

    return df

# def convert_to_clean_df(data_dic):
#     """
#     Preprocess the text, including finding opinion-level data and replacing citations with "CITE".
#     """
#     new_l = []
#     for case in data_dic:
#         case = json.loads(case)
#         soup = BeautifulSoup(case["casebody"]["data"], "html.parser")
#         try:
#             case["opinion_type"] = soup.find("opinion")["type"]
#         except:
#             case["opinion_type"] = "NA"
#         try:
#             case["opinion_author_id"] = soup.find("author")["id"]
#         except:
#             case["opinion_author_id"] = "NA"
#         try:
#             case["opinion_author_text"] = soup.find("author").text
#         except:
#             case["opinion_author_text"] = "NA"
#         try:
#             opinion_text = soup.find("opinion").text
#         except:
#             opinion_text = "NA"
#         opinion_text = find_and_replace_citations(opinion_text)
#         case["pretty_text"] = clean_text(opinion_text)
#         case["word_count"] = len(word_tokenize(case["pretty_text"]))
#         new_l.append(case) 
#     df = pd.DataFrame(new_l)
#     df["close_stat_interp"] = df.apply(lambda x: check_for_close_substrings(x["pretty_text"]), axis = 1)
#     return df

def split_to_paragraphs(df):
    """
    Split at the paragraph level and keep only paragraphs greater than 200 characters long. 
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

def main():
    data = load_file('filtered_data.jsonl')
    longer_cases_df = convert_to_clean_df(data)
    paragraph_df = split_to_paragraphs(longer_cases_df)

    paragraph_df["formal_words"] = paragraph_df.apply(count_formal_seeds, axis = 1)
    paragraph_df["grand_words"] = paragraph_df.apply(count_grand_seeds, axis = 1)
    paragraph_df["grand_count"] = paragraph_df.apply(lambda x: len(x["grand_words"]), axis = 1)
    paragraph_df["formal_count"] = paragraph_df.apply(lambda x: len(x["formal_words"]), axis = 1)

    paragraph_df["diff"] = paragraph_df["formal_count"] - paragraph_df["grand_count"] 
    paragraph_df["diff_percent"] = paragraph_df["diff"] / paragraph_df["par_word_count"]
    
    print("Saving data.")
    paragraph_df.to_csv(os.path.join(os.getcwd(), 'data', 'data_for_predictions.csv'))

    return paragraph_df

if __name__ == '__main__':
    main()
