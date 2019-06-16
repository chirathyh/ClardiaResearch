# Clardia by Chirath Hettiarachchi 
# ANOVA test for Feature Selection
# May 2019

import sys
sys.path.insert(0, '/Users/chirathhettiarachchi/tensorflow/clardia/preprocess')
import experiments as exp
import pandas as pd
import json
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns

def anova_test(x,y):
    print("ANOVA TEST")
    anova = f_classif(x,y)
    result = pd.DataFrame(list(anova))
    head = list(x.columns) 
    result.columns = head
    result.index = ['F','P'] # Row 1 - F values, Row 2 - P values. 

    selected_features = []
    for column in result:
        p = result.loc['P',column]
        if p < 0.05:
            selected_features.append(column)
    #print(selected_features)
    return selected_features

print("\nANOVA: Diabetes vs Healthy ")
x,y = exp.diabetes_featureSelect()
diabetes_features = anova_test(x,y)

df = pd.concat([x, y], axis=1, sort=False)
f, (ax1) = plt.subplots(1, 1, figsize=(24,20))
corr = df.corr() # Entire DataFrame
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Correlation Matrix for Features between Diabetes Only vs Healthy Individuals", fontsize=14)
#plt.show()

# insert box plots
f, axes = plt.subplots(ncols=5, figsize=(20,4))

sns.boxplot(x="Label", y="Age", data=df, ax=axes[0])
axes[0].set_title('Age vs Class ')

sns.boxplot(x="Label", y="apg4", data=df, ax=axes[1])
axes[1].set_title('apg4 vs Class ')

sns.boxplot(x="Label", y="PI_Sys", data=df, showfliers=False,ax=axes[2])
axes[2].set_title('PI_Sys vs Class ')

sns.boxplot(x="Label", y="AI", data=df, showfliers=False,ax=axes[3])
axes[3].set_title('AI vs Class ')

sns.boxplot(x="Label", y="adj_AI", data=df,showfliers=False, ax=axes[4])
axes[4].set_title('adj_AI vs Class ')

plt.show()
##########

'''
print("\nANOVA: Hypertension vs Healthy ")
x,y = exp.hypertension_featureSelect()
hypertension_features = anova_test(x,y)

selected_features = {'diabetes': diabetes_features, 'hypertension': hypertension_features}
print(selected_features)
with open('../cache/selected_features.json', 'w') as fp:
    json.dump(selected_features, fp)
'''
