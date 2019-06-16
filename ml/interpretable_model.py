# Clardia by Chirath Hettiarachchi 
# Interpreting the Results through a Decision Tree
# May 2019

import sys
sys.path.insert(0, '/Users/chirathhettiarachchi/tensorflow/clardia/preprocess')
import experiments as exp
import models as model
import json

import graphviz 
from sklearn import tree

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Read selected features
with open('../cache/selected_features.json') as f:
    data = json.load(f)
features = data['diabetes']
print("Selected Features: ",features)

x,y = exp.experiment_3(features)

cutoff = x.shape[0] - 30
valx   = x[cutoff:]
valy   = y[cutoff:]
trainX = x[:cutoff]
trainY = y[:cutoff]

temp = model.DecisionTree(valx,valy, trainX, trainY, 4)






