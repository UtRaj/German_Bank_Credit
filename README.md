# Overview
This is an end-to-end ML project , which aims at developing a binary classification model for predicting The South German Bank Credit Risk Using The Bank Data into classifying whether it is Good or Bad. 

The final classifier used for this project is Gradient Boosting classifier. 


# Motivation
Normally, most of the bank's wealth is obtained from providing credit loans so that a marketing bank must be able to reduce the risk of non-performing credit loans. The risk of providing loans can be minimized by studying patterns from existing lending data.

One technique that you can use to solve this problem is to use data mining techniques. Data mining makes it possible to find hidden information from large data sets by way of
classification.

The goal of this project, you have to build a model to predict whether the person, described by the attributes of the dataset, is a good (1) or a bad (0) credit risk.


# Dataset Information
This dataset is taken from the UCI Machine Learning Repository. It contains information on defaults, demographic factors, credit data etc. of customers.

Link : (https://archive.ics.uci.edu/ml/datasets/South+German+Credit)


# Approach
1. Data Gathering                          : First I gathered data and made a csv file to be loaded in notebook.

2. Exploratory Data Analysis               : Then ,I visualized data and tried to draw inferences using graphs about various variables.

3. Feature Extraction                      :  Performed Label Encoding and chi-square test.

4. Feature Transformation                  :  Performed Standardization.
                       
5. Initial Model Training and Evaluation   :  Performed Model Training on various models.

6. Hyperparameter Tuning                   : Performed Stacking classifier.

6. Pickle File                             :  Selected model as per best accuracy and created pickle file using Pickle .

7. Webpage & deployment                    :  1. Created a web form that takes all the necessary inputs from user and  shows output.
                                              2. We had to locally deploy in Flask.
                      
