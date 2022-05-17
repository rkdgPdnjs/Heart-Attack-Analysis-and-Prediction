# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:33:52 2022

@author: Alfiqmal
"""

# Train Scripts


#%% Packages 

import os
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")



#%% Paths

# =============================================================================
# ROBUST PATH FOR DATASET, SCALER AND MODEL
# =============================================================================

DATA_PATH = os.path.join(os.getcwd(), "heart.csv")
MMS_PATH = os.path.join(os.getcwd(), "scaler and model", "mms_scaler.pkl")
MODEL_PATH = os.path.join(os.getcwd(), "scaler and model", "model.pkl" )

#%% Data Loading

# =============================================================================
# LOADING DATA UNDER df
# =============================================================================

df = pd.read_csv(DATA_PATH)


#%% Data Intepretation

# =============================================================================
# PRINT THE FIRST 5 ROWS OF OUR DATASET
# =============================================================================

print(df.head(5))

# =============================================================================
# FINDING OUT INFOS ABOUT OUR DATASET
# =============================================================================

print(df.info())

# =============================================================================
# 2 TYPES OF FEATURES, ONE IS CATEGORICAL AND THE OTHER IS CONTINUOUS.
# WE HAVE TO SEPERATE THEM IN OTHER TO SCALE THEM CORRECTLY
# =============================================================================

categorical_column = ["sex", "cp", "fbs", "restecg", "slp", "caa", "exng", 
                      "thall"]
continuous_column = ["age", "trtbps", "chol", "thalachh", "oldpeak"]


#%% Data Cleaning

# =============================================================================
# LOOKING FOR NA AND UNWANTED ZEROES
# =============================================================================

df.isna().sum()
df.isnull().sum()

# no cleaning needed because the data is very well cleaned

#%% Data Preprocessing

df_copy = df

# =============================================================================
# INITIALIZING X AND y
# =============================================================================

X = df_copy.drop(["output"], axis = 1)
y = df_copy[["output"]]

# =============================================================================
# FOR THIS KIND OF DATASET (MEDICAL DATA), IT IS GOOD TO IMPLEMENT MINMAX SCALER
# =============================================================================

mms_scaler = MinMaxScaler()

# =============================================================================
# SAVE OUR SCALER WITH .pkl EXTENSION
# =============================================================================

pickle.dump(mms_scaler, open(MMS_PATH, "wb"))

# =============================================================================
# MINMAX SCALE OUR CONTINUOUS COLUMNS, CATEGORICAL DATA IS GOOD.
# =============================================================================

X[continuous_column] = mms_scaler.fit_transform(X[continuous_column])

# =============================================================================
# TRAIN TEST SPLIT FOR OUR SCALED DATA 
#
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 13)

#%% Machine Learning Pipeline

# =============================================================================
# WE WILL BE COMPARING 3 TYPES OF MACHINE LEARNING, WE WILL CHOOSE THE BEST 
# ONE AFTER WE HAVE OBTAINED THE ACCURACY.
# =============================================================================

steps_RF = [("Random Forest Classifier", RandomForestClassifier())]
steps_DT = [("Decision Tree", DecisionTreeClassifier())]
steps_LR = [("Logistic Regression", LogisticRegression(solver = "liblinear"))]

pipeline_RF = Pipeline(steps_RF)
pipeline_DT = Pipeline(steps_DT)
pipeline_LR = Pipeline(steps_LR)

pipelines = [pipeline_RF, pipeline_DT, pipeline_LR]

for pipe in pipelines:
    pipe.fit(X_train, y_train)

best_r_square = 0
best_pipeline = " "

pipeline_dict = {0: "Random Forest Classifier",
                 1: "Decision Tree Classifier",
                 2: "Logistic Regression"}

for index, model in enumerate(pipelines):
    print(model.score(X_test, y_test))
    if model.score(X_test, y_test) > best_r_square:
        best_r_square = model.score(X_test, y_test)
        best_pipeline = index
        
print("Best Model is {} with accuracy of {}".
      format(pipeline_dict[best_pipeline],best_r_square))


# =============================================================================
# Best Model is Logistic Regression with accuracy of 0.8241758241758241
# =============================================================================

# =============================================================================
# SO WE WILL BE USING LOGISTIC REGRESSION TO OUR SAVED MODEL
# =============================================================================


#%% Parameter Tuning

clf = LogisticRegression()

# =============================================================================
# WE WILL TUNE THE C HYPERPARAMETER TO GET A BETTER ACCURACY
# =============================================================================

parameter_grid = {"C": [0.01, 0.1, 1, 2, 5, 10, 20, 30, 40, 50, 100]}

gridsearch = GridSearchCV(clf, parameter_grid)
gridsearch.fit(X_train, y_train)

print(gridsearch.best_params_)

# =============================================================================
# OUR BEST VALUE {'C': 20}
# =============================================================================



#%% Best Model Save

# =============================================================================
# SAVING OUR MODEL WITH HYPERPARAMETER ADJUSTED, C = 20
# =============================================================================

best_model = LogisticRegression(C = 20, solver = "liblinear")
best_model.fit(X_train, y_train)

pickle.dump(best_model, open(MODEL_PATH, "wb"))


#%% COnfusion Matrix

pred = best_model.predict(X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

