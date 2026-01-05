# Amazon Review Sentiment Analysis

Overview:


Included in this repository are the following items:

1. Preprocessing Data Analysis.ipynb
   - This document contains code for the preprocessing data analysis with the hugging face dataset.
   - The dataset is explored and multiple images are generated.
2. Final Project Models.ipynb
   - This document contains the main codebase for the final project.
   - The dataset is preprocessed and 3 machine learning models are trained and saved based on the dataset.
   - The sentiment_prediction function combines the predictions for each model into one final prediction.
   - A csv file is created based on applying the sentiment_prediction function to a portion of the data
3. Full Model Data Analysis.ipynb
   - This document contains code for exploring the csv file made in document 2.
   - Multiple metrics, tables, and graphs are made and some are saved.
4. Images Folder
   - This folder contains the graphs made in documents 1-3.
5. svm_vectorizer.pkl
   - Pickled version of the SVM vectorizer from document 2.
6. svm_model.pkl
   - Pickled version of the SVM model from document 2.
7. lg_model.pkl
   - Pickled version of the Logistic Regression model and vectorizer from document 2.
8. trained_model folder *(not included)
   - This folder contains the trained model and tokenizer for the BERT model from document 2.
9. logs folder *(not included)
   - This folder contains the logs from training the BERT model in document 2.
10. results folder *(not included)
    - This folder containes the results from training the BERT model in document 2.
11. conclusion_data.csv
    - This is the csv file created in document 2 and used in document 3.
   
References:
- https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/training.ipynb
- https://discuss.huggingface.co/t/evaluating-finetuned-bert-model-for-sequence-classification/5265
- https://medium.com/@nirajan.acharya777/sentimental-analysis-using-linear-regression-86764bfde907
- Other HuggingFace Notebooks
- StackOverflow forums
- Class notes

Notes:
- Items 8-10 are not included as they were too large to upload. these documents can be gained by running the code in documents 2
- The analysis done in document 3 does not require the missing items 8-10
