# legal-interpretation

1. Filter cases to only those decided on or after 1870.
`python filter_cases.py`

2. Collect samples of data for annotation.
`python create_samples.py --output_name annotation_samples --size_sample`
    parser.add_argument('--output_name', type = str)
    parser.add_argument('--size_sample', type = int)

3. Evaluate model performance.
TODO: change to testing folder
`pyhon model_testing.py`

4. Prepare data for predictions at the paragraph level.
`python prep_predictions.py`

5. Run predictions on all opinion paragraphs. 
`python predictions.py`

6. Divide metadata and predictions data. 
`python split_predictions.py`