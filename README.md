# legal-reasoning

1. Filter cases to only those decided by the US Supreme Court on or after 1870.  
`python filter_cases.py`

2. Collect samples of data for annotation.  
`python create_samples.py --output_name annotation_samples --size_sample`

3. Evaluate model performance.  
A number of files within the `model-testing` folder contain all model testing scripts. 

4. Prepare data for predictions at the paragraph level.  
`python prep_predictions.py`

5. Run predictions on all opinion paragraphs.   
`python predictions.py`

6. Divide metadata and predictions data.   
`python split_predictions.py`

Email ret85@cornell.edu to get access to the annotated data. The data will be posted publicly at a later date. 
