import pandas as pd
import os
from collections import Counter

output_dir = os.path.join(os.getcwd(), 'predictions')
predictions_dir = os.path.join(output_dir, 'combined_output.csv')
df = pd.read_csv(predictions_dir)

predictions_df = df[["id", "section_id", "paragraph", "grand_count", "formal_count", "predictions", "prob_0", "prob_1", "prob_2", "predicted_class"]]
metadata_df = df[["id", "url", "name", "name_abbreviation", "decision_date", "opinion_type","opinion_author_id","opinion_author_text","docket_number", "citations", "volume", "reporter","court","jurisdiction","cites_to","frontend_url"]]
metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'))
predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'))
