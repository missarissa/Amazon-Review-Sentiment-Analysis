# Amazon Review Sentiment Analysis

Overview:
This repository contains the full workflow for a sentiment analysis pipeline built utilizing SVM, BERT, and Logistic Regression. The project explores text data preprocessing (STOP words, vectorization, etc.), model training, ensemble-style prediction, and model evaluation to compare across the 3 model methods.

The goal of this project is to demonstrate:
   - End-to-end NLP model development with lower level tasks
   - Comparison between classic ML and transformer ML models

Repository Structure:
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
8. conclusion_data.csv
    - This is the csv file created in document 2 and used in document 3.

Discluded Items:
1. trained_model folder
   - This folder contains the trained model and tokenizer for the BERT model from document 2.
2. logs folder
   - This folder contains the logs from training the BERT model in document 2.
3. results folder
    - This folder containes the results from training the BERT model in document 2.
     
References:
- https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/training.ipynb
- https://discuss.huggingface.co/t/evaluating-finetuned-bert-model-for-sequence-classification/5265
- https://medium.com/@nirajan.acharya777/sentimental-analysis-using-linear-regression-86764bfde907
- Other HuggingFace Notebooks
- StackOverflow forums
