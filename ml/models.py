# Clardia - Diabetes Prediction by Chirath Hettiarachchi
# Classification Models, Hyperparameter Tuning, Prediction
# May 2019

import numpy as np
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint

from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import cross_validate, cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# todo -> add the grid search capability, option to choose between kfold, stratified
# Nested CV 

class Model:
    def __init__(self, x, y, testx, testy, folds):
        self.x = x
        self.y = y
        self.testx = testx
        self.testy = testy
        self.skf = StratifiedKFold(n_splits=folds, shuffle=False)

    def tune_hyperparams(self):
        n_iter_search = 50
        self.tuned = RandomizedSearchCV(estimator=self.clf, 
                    param_distributions=self.p_grid, 
                    cv=self.skf,n_iter=n_iter_search,
                    scoring='roc_auc', iid=False)
        self.tuned.fit(self.x, self.y)
        self.tuned_params= self.tuned.best_params_
        #print(self.tuned.best_score_)
        print("The tuned Hyperparameters: ",self.tuned_params)

    def train(self):
        #self.tuned.fit(self.testx, self.testy)
        #temp = cross_val_score(self.tuned, self.x, self.y, cv=self.skf)
        crossValResults =  cross_validate(self.tuned, self.testx, self.testy, 
                          cv=self.skf ,scoring = ['roc_auc','f1','precision','recall'])
        f1        = crossValResults['test_f1']
        roc_auc   = crossValResults['test_roc_auc']
        precision = crossValResults['test_precision']
        recall    = crossValResults['test_recall']
        #print(sorted(crossValResults.keys()))
        print("Mean ROC_AUC  :%f   SD: %f " %(np.mean(roc_auc),np.std(roc_auc)))
        print("Mean F1       :%f   SD: %f " %(np.mean(f1),np.std(f1)))
        print("Mean Precision:%f   SD: %f " %(np.mean(precision),np.std(precision)))
        print("Mean Recall   :%f   SD: %f " %(np.mean(recall),np.std(recall)))
        #print('\nThe raw scores')
        #print(f1)
        #print(roc_auc)

class LDA(Model):
    def __init__(self, x, y, testx, testy, folds):
        Model.__init__(self, x, y, testx, testy,folds)
        print("Linear Discriminant Analysis")
        self.clf = LinearDiscriminantAnalysis()
        self.set_hyperparameters()
        self.tune_hyperparams()
        self.train()

    def set_hyperparameters(self):
        self.p_grid = {'solver': ['svd','lsqr','eigen']}       

class DecisionTree(Model):
    def __init__(self, x, y, testx, testy, folds):
        Model.__init__(self, x, y, testx, testy, folds)
        print("Decision Tree Classifier")
        self.clf = tree.DecisionTreeClassifier()
        self.set_hyperparameters()
        self.tune_hyperparams()
        self.train()

    def set_hyperparameters(self):
        self.p_grid = {'criterion': ['gini','entropy'],
                        'max_depth': sp_randint(1, 4),
                        'min_samples_split': sp_randint(2, 10),
                        'min_samples_leaf': sp_randint(1, 10)}      

class KNN(Model):
    def __init__(self, x, y, testx, testy, folds):
        Model.__init__(self, x, y, testx, testy, folds)
        print("KNN Classifier")
        self.clf = KNeighborsClassifier(n_neighbors=2,algorithm='brute')
        self.set_hyperparameters()
        self.tune_hyperparams()
        self.train()

    def set_hyperparameters(self):
        self.p_grid = {'weights': ['uniform','distance'],'p': [1,2]}

class RandomForest(Model):
    def __init__(self, x, y, testx, testy, folds):
        Model.__init__(self, x, y, testx, testy, folds)
        print("Random Forest Classifier")
        self.clf = RandomForestClassifier(n_jobs =-1)
        self.set_hyperparameters()
        self.tune_hyperparams()
        self.train()

    def set_hyperparameters(self):
        self.p_grid = {'min_samples_split':  sp_randint(2, 10),
                        'n_estimators': sp_randint(1, 300),
                        'max_depth':  sp_randint(1, 4)}

class AdaBoost(Model):
    def __init__(self, x, y, testx, testy, folds):
        Model.__init__(self, x, y, testx, testy, folds)
        print("Adaboost Classifier")
        self.clf = AdaBoostClassifier()
        self.set_hyperparameters()
        self.tune_hyperparams()
        self.train()

    def set_hyperparameters(self):
        self.p_grid = {'n_estimators': sp_randint(1, 300),
                        'learning_rate': sp_rand(),
                        'algorithm': ['SAMME', 'SAMME.R']}

class LogisicRegression(Model):
    def __init__(self, x, y, testx, testy, folds):
        Model.__init__(self, x, y, testx, testy, folds)
        print("Logistic Regression Classifier")
        self.clf = LogisticRegression()
        self.set_hyperparameters()
        self.tune_hyperparams()
        self.train()

    def set_hyperparameters(self):
        self.p_grid = {'C': sp_rand(),
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear','saga'],
                        'max_iter': sp_randint(1,3000)}

class SVM_Linear(Model):
    def __init__(self, x, y, testx, testy, folds):
        Model.__init__(self, x, y, testx, testy, folds)
        print("SVM  Classifier")
        self.clf = SVC(kernel='linear')
        self.set_hyperparameters()
        self.tune_hyperparams()
        self.train()

    def set_hyperparameters(self):
        self.p_grid = {'C': sp_rand()}

class NaiveBayes(Model):
    def __init__(self, x, y, testx, testy, folds):
        Model.__init__(self, x, y, testx, testy, folds)
        print("NaiveBayes  Classifier")
        self.clf = GaussianNB()
        self.tuned = self.clf
        self.train()


