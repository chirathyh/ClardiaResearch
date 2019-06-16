# Clardia by Chirath Hettiarachchi 
# Statistical Interpretation of the ANOVA features
# May 2019

# todo -> calculate the means std dev of the identifies features. 

import sys
sys.path.insert(0, '/Users/chirathhettiarachchi/tensorflow/clardia/preprocess')
import experiments as exp
import pandas as pd
import json

# Read selected features
with open('../cache/selected_features.json') as f:
    data = json.load(f)
features = data['diabetes']
print("Selected Features: ",features)

x,y = exp.diabetes_featureSelect()
df = pd.concat([x,y], axis=1)
features.append('Label')
df = df[features]
#print(df)

diabetes = df.loc[df['Label'] == 1]
healthy = df.loc[df['Label'] == 0]

features.remove('Label')
for column in features:
    print("Analyzing Feature: ", column)
    print("\nHealthy Values")
    print("Mean  : ", healthy[column].mean())
    print("StdDev: ", healthy[column].std())
    print("\nDiabetes Values")
    print("Mean  : ", diabetes[column].mean())
    print("StdDev: ", diabetes[column].std())
    print("\n")
