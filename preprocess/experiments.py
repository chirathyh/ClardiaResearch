# Clardia by Chirath Hettiarachchi 
# Script to define experiments (Manage Data)
# May 2019

import numpy as np  
from numpy.random import seed
seed(7)
import pandas as pd
from sklearn.utils import shuffle

def import_data(experiment):
    # Function to prepare the data for the target experiment. 
    
    # Diabetes related Data
    df1 = pd.read_csv('../input/diabetesFeatures.csv', sep=',', header = 0)
    df1 = df1.dropna()
    df3 = df1.loc[(df1['Hypertension'] == 0)] # Only Diabetes: label 1
    df4 = df1.loc[(df1['Hypertension'] == 1)] # Diabetes + Prehypertension: label 2
    df5 = df1.loc[(df1['Hypertension'] == 2) | (df1['Hypertension'] == 3)] # Diab+Hyper: label 3

    #Normal
    df2 = pd.read_csv('../input/normalFeatures.csv', sep=',', header = 0)
    df2 = df2.dropna()
    df2['Hypertension'] = 0

    #Hypertension 
    df6 = pd.read_csv('../input/hypertensionFeatures.csv', sep=',', header = 0)
    df6 = df6.dropna()

    if experiment == 1:
        print("Imported Data for Experiment 1")
        return df2, df3
    elif experiment == 2:
        print("Imported Data for Experiment 2")
        return df2, df3, df4
    elif experiment == 3:
        print("Imported Data for Experiment 3")
        return df2, df3, df4, df5
    elif experiment == 4:
        print("Imported Data for Experiment 4")
        return df2, df6
    else:
        print("ERROR: Invalid Experiment")

def experiment_1 (features):
    # Analyzing healthy Vs Only Diabetes
    print("Experiment 1: Analyzing healthy Vs Only Diabetes ")
    df2, df3 = import_data(experiment=1)   
    df2 = df2.sample(n=9)
    df = pd.concat([df2, df3], axis=0, sort=False)
    df['apg9'] = df['apg3'] * -1
    #male = 1, BF =  1.2 * BMI + 0.23 * Age -10.8 * Gender - 5.4
    df['BodyFat'] = (1.2 * df['BMI']) + (0.23 * df['Age']) - (10.8 * df['Gender']) - 5.4
    df = shuffle(df, random_state=7)
    y = df['Label']
    y = y.replace(2, 1) #convert y=2 vaiable to y=1
    x = df[features]
    return x,y

def experiment_2 (features):
    # Analyzing halthy , diabetes (only,pre)
    print("Experiment 2: Analyzing halthy , diabetes (only,pre)")
    df2, df3, df4 = import_data(experiment=2)  
    df2 = df2.sample(n=25)
    df = pd.concat([df2, df3, df4], axis=0, sort=False)
    df['apg9'] = df['apg3'] * -1
    df['BodyFat'] = (1.2 * df['BMI']) + (0.23 * df['Age']) - (10.8 * df['Gender']) - 5.4
    df = shuffle(df, random_state=7)
    y = df['Label']
    y = y.replace(2, 1)
    x = df[features]
    return x,y 

def experiment_3 (features):
    # Analyzing healthy , diabetes (only,pre, hyper)
    print("Experiment 3: Analyzing healthy , diabetes (only,pre, hyper)")
    df2, df3, df4, df5 = import_data(experiment=3) 
    df2 = df2.sample(n=32)
    df = pd.concat([df2, df3, df4, df5], axis=0, sort=False)
    df['apg9'] = df['apg3'] * -1
    df['BodyFat'] = (1.2 * df['BMI']) + (0.23 * df['Age']) - (10.8 * df['Gender']) - 5.4
    df = shuffle(df, random_state=7)
    y = df['Label']
    y = y.replace(2, 1)
    x = df[features]
    return x,y  

def hypertension_featureSelect():
    # Analyze Hypertension only with healthy
    print("Dataset for Hypertension Feature Select")
    df1, df2 = import_data(experiment=4) 
    df = pd.concat([df1, df2], axis=0, sort=False)
    df['apg9'] = df['apg3'] * -1
    df['BodyFat'] = (1.2 * df['BMI']) + (0.23 * df['Age']) - (10.8 * df['Gender']) - 5.4
    df = shuffle(df, random_state=7)
    y = df['Hypertension']
    y = y.replace(3, 2) #treat stage 1 & 2 hypertension together
    x = df.drop(['id','Label','Hypertension'], axis = 1)
    return x,y

def diabetes_featureSelect():
    # Analyzing healthy Vs Only Diabetes
    print("Dataset for Diabetes Feature Select")
    df2, df3 = import_data(experiment=1)   
    df = pd.concat([df2, df3], axis=0, sort=False)
    df['apg9'] = df['apg3'] * -1
    df['BodyFat'] = (1.2 * df['BMI']) + (0.23 * df['Age']) - (10.8 * df['Gender']) - 5.4
    df = shuffle(df, random_state=7)
    y = df['Label']
    y = y.replace(2, 1) #convert y=2 vaiable to y=1
    x = df.drop(['id','Label','Hypertension'], axis = 1)
    return x,y


# Male : 1
# Female : 0
# Diabetes Normal : 0
# Diabetes : 1
# Type 2 Diabetes : 2
# Hypertension Normal : 0
# Prehypertension : 1
# Stage 1 hypertension : 2
# Stage 2 hypertension : 3

# Test
#features = ['Age','apg4','PI_Sys','AI','adj_AI']
#x,y = experiment_4()
#print(x)
#print(y)