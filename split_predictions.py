import pandas as pd
import os
from collections import Counter

output_dir = os.path.join(os.getcwd(), 'predictions')
predictions_dir = os.path.join(output_dir, 'combined_output.csv')
df = pd.read_csv(predictions_dir)


predictions_df = df[["section_id", "case_id", "proj_case_id", "proj_opinion_id",\
                      "proj_id", "paragraph", "grand_count", \
                     "formal_count", "predictions", "prob_0", "prob_1",\
                          "prob_2", "predicted_class"]]

metadata_df = df[["case_id", "proj_case_id", "proj_opinion_id", "proj_id", \
                  "case_url", "case_name", "citations", "date", "court_url", \
                    "opinion_type", "opinion_text", "close_stat_interp", \
                        "authors_raw", "author_names", "author_ids"]]

metadata_df = metadata_df.drop(columns={"Unnamed: 0"}).drop_duplicates()

metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'))
predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'))


