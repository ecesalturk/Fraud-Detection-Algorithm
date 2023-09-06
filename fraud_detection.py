# -*- coding: utf-8 -*-
"""
Created on 18 June 2022

@author: Ece
"""
#set up: importing libaries
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score

from sklearn import metrics
from collections import Counter

#load the csv file as a data frame
rawdata_df = pd.read_csv(("/Users/ecesalturk/Desktop/creditdata/creditcard.csv"))[0:12000]

#data imbalance visually through a pi chart
#labels=["Legit","Fraud"]
fraud_or_not=rawdata_df["Class"].value_counts().tolist()
values=[fraud_or_not[0], fraud_or_not[1]]
#figure= px.pie(values=rawdata_df["Class"].value_counts(), names=labels, width=700, height=700, color_discrete_sequence=["skyblue", "black"], title="Fraud vs Legit Transactions")
#figure.show()

#data imbalance visually through bar chart
plt.figure(figsize=(3,4))
ax = sns.countplot(x="Class",data=rawdata_df,palette="pastel")
for i in ax.containers:
    ax.bar_label(i,)

print("Legit:", round(rawdata_df["Class"].value_counts()[0]/len(rawdata_df) * 100,2), "% of the dataset")
print("Fraud:", round(rawdata_df["Class"].value_counts()[1]/len(rawdata_df) * 100,2), "% of the dataset")

#dropping duplicated values
df = rawdata_df.copy()
df.drop_duplicates(inplace=True)
print("Duplicated values dropped succesfully")
print("*" * 100)

df=df.drop("Time", axis=1)

#outlier check- we don't remove any
numeric_columns=(list(df.loc[:,"V1":"Amount"]))

# checking boxplots
def boxplots_custom(dataset, columns_list, rows, cols, suptitle):
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(16,25))
    fig.suptitle(suptitle,y=1, size=25)
    axs = axs.flatten()
    for i, data in enumerate(columns_list):
        sns.boxplot(data=dataset[data], orient='h', ax=axs[i])
        axs[i].set_title(data + ', skewness is: '+str(round(dataset[data].skew(axis = 0, skipna = True),2)))
        
boxplots_custom(dataset=df, columns_list=numeric_columns, rows=10, cols=3, suptitle='Boxplots for each variable')
plt.tight_layout()

#Using Tukey's IQR method  to detect outliers
def IQR_method (df, n, features):
    outlierlist=[]
    for column in features:
        #quartile no1 (25%)
        Q1=np.percentile(df[column], 25)
        #quartile no3 (75%)
        Q3=np.percentile(df[column], 75)
        #Interquartile range
        IQR=Q3-Q1
        #outlier step
        outlier_step=1.5*IQR
        #Determining the list of indices of outliers
        outlierlist_column = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step )].index
        outlierlist.extend(outlierlist_column)
    #selecting observations with more than x outliers
    outlierlist=Counter(outlierlist)
    multiple_outliers=list(k for k, v in outlierlist.items() if v>n)
    #calculate the number of records beyond the accepted range
    out_of_range1=df[df[column]<Q1-outlier_step]
    out_of_range2=df[df[column]>Q3+outlier_step]
    print("Total number of excluded outliers is:", out_of_range1.shape[0]+out_of_range2.shape[0])
    return multiple_outliers
#detecting outliers
outliers_IQR=IQR_method(df, 1, numeric_columns)
#dropping outliers
df_dropped=df.drop(outliers_IQR, axis=0).reset_index(drop=True)

#plotting again to see the results(without outliers)
plt.figure(figsize=(3,4))
ax=sns.countplot(x="Class", data=df_dropped, palette="pastel")
for i in ax.containers:
    ax.bar_label(i,)    

#stratified splitting: we split because we will keep test untouched while under/oversampling the train set
A=df.drop("Class", axis=1)
B=df["Class"]
from sklearn.model_selection import train_test_split
A_train, A_test, B_train, B_test= train_test_split(A, B, stratify=B, test_size=0.3, random_state=42)

#feature scaling
from sklearn.preprocessing import StandardScaler

#Creating a function for scaling
def Standard_Scaler (df, col_names):
    features=df[col_names]
    scaler=StandardScaler().fit(features.values)
    features=scaler.transform(features.values)
    df[col_names]=features
    return df
col_names=["Amount"]
A_train=Standard_Scaler(A_train, col_names)
A_test=Standard_Scaler(A_test, col_names)

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
#to ensure we have the same splits of data every time we create a KFold object (kf) and pass cross validation=kf
kf=StratifiedKFold(n_splits=5, shuffle=False)
rf=RandomForestClassifier(n_estimators=100, random_state=13)

#keep in mind the focus is on false negatives(fraud that are considered as legit transactions)
score = cross_val_score(rf, A_train, B_train, cv=kf, scoring="recall")
print("Cross Validation Recall scores are: {}".format(score))
print("Average Cross Validation Recall score: {}".format(score.mean()))

#hyperparameter tunining using GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {
    "n_estimators": [50, 100, 200],
    "max_depth": [4, 6, 10, 12],
    "random_state": [13]
}
grid_rf=GridSearchCV(rf, param_grid=parameters, cv=kf, scoring="recall").fit(A_train, B_train)
print("Best Parameters:", grid_rf.best_params_)
print("Best Score:", grid_rf.best_score_)

#pre-sampling the precision was 77%, to compare:
B_pred=grid_rf.predict(A_test)
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
cm=confusion_matrix(B_test, B_pred)
rf_Recall=recall_score(B_test, B_pred)
rf_Precision=precision_score(B_test, B_pred)
rf_f1=f1_score(B_test, B_pred)
rf_accuracy=accuracy_score(B_test, B_pred)
print(cm)
#these will be the scores for a test set
ndf=[(rf_Recall, rf_Precision, rf_f1, rf_accuracy)]
rf_score=pd.DataFrame(data=ndf, columns=["Recall", "Precision", "F1 Score", "Accuracy"])
rf_score.insert(0, "Random Forest with", "No Under/Oversampling")
rf_score

#oversampling
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=42)
#applying the transform
A_over, B_over =ros.fit_resample(A_train, B_train)
print("Legit:", B_over.value_counts()[0], "/", round(B_over.value_counts()[0]/len(B_over)*100,2), "% of the dataset")
print("Frauds:", B_over.value_counts()[1], "/", round(B_over.value_counts()[1]/len(B_over)*100,2), "% of the dataset")

#pipeline
from imblearn.pipeline import Pipeline, make_pipeline
random_overs_pipeline = make_pipeline(RandomOverSampler(random_state=42), RandomForestClassifier(n_estimators=100, random_state=13))
second_score=cross_val_score(random_overs_pipeline, A_train, B_train, scoring="recall", cv=kf)
print("Cross Validation Recall Scores are: {}".format(second_score))
print("Average Cross Validation Recall score: {}".format(second_score.mean()))

#GridSearchCrossValidation
new_parameters={"randomforestclassifier__" + key: parameters[key] for key in parameters}
grid_over_rf = GridSearchCV(random_overs_pipeline, param_grid=new_parameters, cv=kf, scoring="recall", return_train_score=True)
grid_over_rf.fit(A_train, B_train)
print("Best Parameters:", grid_over_rf.best_params_)
print("Best Score:", grid_over_rf.best_score_)

#confusion matrix and scores
B_pred = grid_over_rf.best_estimator_.named_steps["randomforestclassifier"].predict(A_test)
cm=confusion_matrix(B_test, B_pred)
over_rf_Recall = recall_score(B_test, B_pred)
over_rf_Precision = precision_score(B_test, B_pred)
over_rf_f1 = f1_score(B_test, B_pred)
over_rf_accuracy = accuracy_score(B_test, B_pred)
print(cm)

ndf=[(over_rf_Recall, over_rf_Precision, over_rf_f1, over_rf_accuracy)]
#creating a data table to see the changes due to our manupilations
over_rf_score = pd.DataFrame(data = ndf, columns=["Recall","Precision","F1 Score", "Accuracy"])
over_rf_score.insert(0, "Random Forest with", "Random Oversampling")
over_rf_score

#random undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
A_under, B_under = rus.fit_resample(A_train, B_train)
print("Legit:", B_under.value_counts()[0], "/", round(B_under.value_counts()[0]/len(B_under) * 100,2), "% of the dataset")
print("Frauds:", B_under.value_counts()[1], "/",round(B_under.value_counts()[1]/len(B_under) * 100,2), "% of the dataset")

#SMOTE
from imblearn.over_sampling import SMOTE
smote_pipeline = make_pipeline(SMOTE(random_state=42), RandomForestClassifier(n_estimators=100, random_state=13))
third_score = cross_val_score(smote_pipeline, A_train, B_train, scoring="recall", cv=kf)
print("Cross Validation Recall Scores are: {}".format(third_score))
print("Average Cross Validation Recall score: {}".format(third_score.mean()))

#Grid Search Cross Validation
new_parameters = {"randomforestclassifier__" + key: parameters[key] for key in parameters}
smote_rf = GridSearchCV(smote_pipeline, param_grid=new_parameters, cv=kf, scoring="recall", return_train_score=True)
smote_rf.fit(A_train, B_train)

print("Best parameters:", smote_rf.best_params_)
print("Best score:", smote_rf.best_score_)

#confusion matrix and results
B_pred = smote_rf.best_estimator_.named_steps["randomforestclassifier"].predict(A_test)
cm = confusion_matrix(B_test, B_pred)
smote_rf_Recall = recall_score(B_test, B_pred)
smote_rf_Precision = precision_score(B_test, B_pred)
smote_rf_f1 = f1_score(B_test, B_pred)
smote_rf_accuracy = accuracy_score(B_test, B_pred)
print(cm)

ndf = [(smote_rf_Recall, smote_rf_Precision, smote_rf_f1, smote_rf_accuracy)]
#constructing a data table to see our results
smote_rf_score = pd.DataFrame(data = ndf, columns=["Recall","Precision",'F1 Score', 'Accuracy'])
smote_rf_score.insert(0, 'Random Forest with', 'SMOTE Oversampling')
smote_rf_score

#Undersampling using Tomek links
from imblearn.under_sampling import TomekLinks
tomekU=TomekLinks()
A_underT, B_underT = tomekU.fit_resample(A_train, B_train)
print("Legit:", B_underT.value_counts()[0], "/", round(B_underT.value_counts()[0]/len(B_underT)*100,2), "% of the dataset.")
print("Fraud:", B_underT.value_counts()[1], "/", round(B_underT.value_counts()[1]/len(B_underT)*100,2), "% of the dataset.")

from imblearn.combine import SMOTETomek
SMOTETomek_pipeline= make_pipeline(SMOTETomek(tomek=TomekLinks(sampling_strategy="majority")), RandomForestClassifier(n_estimators=100, random_state=13))

#GridSearchCV
print(A_train.columns)
# SMOTETomek_rf=SMOTETomek_pipeline
# SMOTETomek_rf.fit(A_train, B_train)

# #confusion matrix and scores
# B_pred=SMOTETomek_rf.predict(A_test)
# cm=confusion_matrix(B_test, B_pred)
# SMOTETomek_rf_Recall=recall_score(B_test, B_pred)
# SMOTETomek_rf_Precision=precision_score(B_test, B_pred)
# SMOTETomek_rf_f1=f1_score(B_test, B_pred)
# SMOTETomek_rf_accuracy=accuracy_score(B_test, B_pred)
# print(cm)

# ndf = [(SMOTETomek_rf_Recall, SMOTETomek_rf_Precision, SMOTETomek_rf_f1, SMOTETomek_rf_accuracy)]
# SMOTETomek_rf_score=pd.DataFrame(data=ndf, columns=["Recall", "Precision", "F1 Score", "Accuracy"])
# SMOTETomek_rf_score.insert(0, "Random Forest with", "SMOTE plus Tomek")
# SMOTETomek_rf_score

# #training model: the classes will be weighted inversely proportional to how frequently they appear among the data set
# rfb = RandomForestClassifier(n_estimators=100, random_state=13, class_weight="balanced")
# score5 = cross_val_score(rfb, A_train, B_train, cv=kf, scoring="recall")
# print("Cross Validation Recall scores are: {}".format(score5))
# print("Average Cross Validation Recall score: {}".format(score5.mean()))

# #GridSearchCrossValidation
# grid_rfb=GridSearchCV(rfb, param_grid=parameters, cv=kf, scoring="recall").fit(A_train, B_train)

# #cm and scores
# B_pred = grid_rfb.predict(A_test)
# cm = confusion_matrix(B_test, B_pred)
# grid_rfb_Recall = recall_score(B_test, B_pred)
# grid_rfb_Precision = precision_score(B_test, B_pred)
# grid_rfb_f1 = f1_score(B_test, B_pred)
# grid_rfb_accuracy = accuracy_score(B_test, B_pred)
# print(cm)

# ndf = [(grid_rfb_Recall, grid_rfb_Precision, grid_rfb_f1, grid_rfb_accuracy)]
# grid_rfb_score = pd.DataFrame(data = ndf, columns=["Recall", "Precision", "F1 Score", "Accuracy"])
# grid_rfb_score.insert(0, "Random Forest with", "class weights")
# grid_rfb_score

# #comparing results
# predictions = pd.concat([rf_score, over_rf_score, smote_rf_score, SMOTETomek_rf_score, grid_rfb_score], ignore_index=True, sort=False)
# predictions.sort_values(by=["Recall"], ascending=False)

# import pickle

# filename = "creditcard.pickle"

# # save model
# pickle.dump(SMOTETomek_rf, open(filename, "wb"))

# # load model
# #loaded_model = pickle.load(open(filename, "rb"))





