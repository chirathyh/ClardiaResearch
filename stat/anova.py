# Clardia by Chirath Hettiarachchi 
# ANOVA test for Feature Selection
# May 2019

import sys
sys.path.insert(0, '/Users/chirathhettiarachchi/tensorflow/clardia/preprocess')
import experiments as exp
import pandas as pd
import json
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif

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

print("\nANOVA: Hypertension vs Healthy ")
x,y = exp.hypertension_featureSelect()
hypertension_features = anova_test(x,y)

selected_features = {'diabetes': diabetes_features, 'hypertension': hypertension_features}
print(selected_features)

with open('../cache/selected_features.json', 'w') as fp:
    json.dump(selected_features, fp)

# todo -> analyzing the selected features, interpretation