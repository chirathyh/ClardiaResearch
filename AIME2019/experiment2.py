# Chirath Hettiarachchi 2019
# Experiment 2: Healthy vs Diabetes Only + Diabetes with Prehypertension.

import pandas as pd
from sklearn.utils import shuffle
import numpy as np
seed = 7
np.random.seed(seed)

from classifiers import decisionTree, LDA, naiveBayesClassifier, randomForest, logisicRegression, adaBoostClassifier, svmClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def Experiment2(df2, df3, df4):
    # Prepare the dataframes for analysis.
    df2 = df2.sample(n=25)
    # Removing younger crowd to test the robustness
    # df2= df2.loc[(df2['Age'] >= 30)]
    # df3= df3.loc[(df3['Age'] >= 30)]
    # df4= df4.loc[(df4['Age'] >= 30)]
    # df2 = df2.sample(n=24)
    df = pd.concat([df2, df3, df4], axis=0, sort=False)
    df['apg9'] = df['apg3'] * -1
    # male = 1, BF =  1.2 * BMI + 0.23 * Age -10.8 * Gender - 5.4
    df['BodyFat'] = (1.2 * df['BMI']) + (0.23 * df['Age']) - (10.8 * df['Gender']) - 5.4
    df = shuffle(df, random_state=7)
    #print df
    y = df['Label']
    y = y.replace(2, 1)  # convert y=2 vaiable to y=1

    # select features should be passed.
    x = df[['Age', 'apg4', 'PI_Sys', 'AI', 'adj_AI']] # features selected from ANOVA.
    # x = df[['apg4','PI_Sys','AI','adj_AI']] #features without Age
    # x = df.drop(['id','Label','Gender','Age','Height','Weight','BMI','Hypertension'], axis = 1) #all PPG features
    # x = df.drop(['id','Label','Hypertension'], axis = 1) #All features.
    return x, y

#load the data.
#Diabetes
df1 = pd.read_csv('input/diabetesAPGFeaturesV2.csv', sep=',', header = 0)
df1 = df1.dropna()
df3 = df1.loc[(df1['Hypertension'] == 0)] # Only Diabetes: label 1
df4 = df1.loc[(df1['Hypertension'] == 1)] # Diabetes + Prehypertension: label 2
df5 = df1.loc[(df1['Hypertension'] == 2) | (df1['Hypertension'] == 3)] # Diabetes + Hypertension: label 3
#Normal
df2 = pd.read_csv('input/normalFullFeatures.csv', sep=',', header = 0)
df2 = df2.dropna()
df2['Hypertension'] = 0
#df2= df2.loc[(df2['Age'] >= 50)]

x,y = Experiment2(df2,df3,df4)
naiveBayesClassifier(x,y,10)
logisicRegression(x,y,10)
adaBoostClassifier(x,y,10)
randomForest(x,y,10)
decisionTree(x,y,10)
svmClassifier(x,y,10)
LDA(x,y,10)



