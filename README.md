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

The annotation data for this project is available [here](https://drive.google.com/file/d/1i7dcshwcgCBF3TVLbNBC-Hutw8qVreTq/view?usp=sharing). If you use this data, please cite [Modeling Legal Reasoning: LM Annotation at the Edge of Human Agreement](https://aclanthology.org/2023.emnlp-main.575.pdf).
