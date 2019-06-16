# Clardia by Chirath Hettiarachchi 
# ANOVA test for Feature Selection
# May 2019

import sys
sys.path.insert(0, '/Users/chirathhettiarachchi/tensorflow/clardia/preprocess')
import experiments as exp
import pandas as pd
import json
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

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

x,y = exp.experiment_3([])
#x,y = exp.diabetes_featureSelect()

#diabetes_features = anova_test(x,y)
#x = x[diabetes_features]
#print(x)

# Normalizing the Input
x = x.values #returns a numpy array
standard_scaler = preprocessing.StandardScaler()
x_scaled = standard_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)



t0 = time.time()
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1 - t0))

# PCA Implementation
t0 = time.time()
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
print("PCA took {:.2} s".format(t1 - t0))

# TruncatedSVD
t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
t1 = time.time()
print("Truncated SVD took {:.2} s".format(t1 - t0))


##########
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)


blue_patch = mpatches.Patch(color='#0A0AFF', label='Healthy')
red_patch = mpatches.Patch(color='#AF0000', label='Diabetes')


# t-SNE scatter plot
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='Healthy', linewidths=2)
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Diabetes', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)

ax1.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='Healthy', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Diabetes', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='Healthy', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Diabetes', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.show()