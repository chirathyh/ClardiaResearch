# Chirath Hettiarachchi
# Classifiers, parameter tuning and evaluation.

import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import make_pipeline
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint

def modelEvaluator(clf, x, y, folds):
    #calculate the metrics of the target models.
    print("Model evaluated through Cross Validation (k=10)")
    skf = StratifiedKFold(folds) # kf = KFold(n_splits=10) # -> when used there can be unbalanced splits
    crossValResults = cross_validate(clf, x, y, cv=skf, scoring=['roc_auc', 'f1', 'precision', 'recall'])
    f1 = crossValResults['test_f1']
    roc_auc = crossValResults['test_roc_auc']
    precision = crossValResults['test_precision']
    recall = crossValResults['test_recall']
    # print(sorted(crossValResults.keys()))
    print("Mean ROC_AUC:%f   SD: %f " % (np.mean(roc_auc), np.std(roc_auc)))
    print("Mean F1:%f   SD: %f " % (np.mean(f1), np.std(f1)))
    print("Mean Precision:%f   SD: %f " % (np.mean(precision), np.std(precision)))
    print("Mean Recall:%f   SD: %f " % (np.mean(recall), np.std(recall)))

    #calculate the specificity, using precision, recall considering the balanced classes.
    specificity = []
    for ii in range(0,len(recall)):
        x = recall[ii]
        y = precision[ii]
        y = max(y,0.0000001) #avoid the div by zero.
        fp = x/y - x
        tn = 1 - fp
        specificity.append(tn)
    print("Mean Specificity:%f   SD: %f " % (np.mean(specificity), np.std(specificity)))
    #print('\nThe raw scores')
    #print f1
    #print roc_auc


def naiveBayesClassifier(x, y, folds):
    print("\nNaiveBays Classifier")
    gnb = GaussianNB()
    modelEvaluator(gnb, x, y, folds)


def adaBoostClassifier(x, y, folds):
    print("\nAda Boost Classifier -Best Tuned Model")

    clf = AdaBoostClassifier(random_state=7)
    pipe = make_pipeline(preprocessing.StandardScaler(), clf)

    skf = StratifiedKFold(folds)
    n_iter_search = 50
    # Hyper parameters
    n_estimators = sp_randint(1, 300)
    learning_rate = sp_rand()
    algorithm = ['SAMME', 'SAMME.R']

    param_dist = dict(adaboostclassifier__n_estimators=n_estimators,
                      adaboostclassifier__learning_rate=learning_rate, adaboostclassifier__algorithm=algorithm)

    random_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search, cv=skf,
                                       scoring='roc_auc', verbose=1)
    random_search.fit(x, y)
    tuned_params = random_search.best_params_
    print tuned_params

    # Tuned Hyper parameters
    n_estimators = tuned_params['adaboostclassifier__n_estimators']
    learning_rate = tuned_params['adaboostclassifier__learning_rate']
    algorithm = tuned_params['adaboostclassifier__algorithm']

    print('\nAB Best Tuned Model')
    tuned_clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm,
                                   random_state=7)
    modelEvaluator(tuned_clf, x, y, folds)


def logisicRegression(x, y, folds):
    n_iter_search = 50
    skf = StratifiedKFold(folds)

    clf = LogisticRegression(random_state=7)
    pipe = make_pipeline(preprocessing.StandardScaler(), clf)

    # Hyper parameters
    C_range = sp_rand()
    penalty_options = ['l1', 'l2']
    solver = ['liblinear', 'saga']
    max_iter = sp_randint(1, 3000)

    param_dist = dict(logisticregression__C=C_range, logisticregression__penalty=penalty_options,
                      logisticregression__solver=solver, logisticregression__max_iter=max_iter)

    random_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search, cv=skf,
                                       scoring='roc_auc')
    random_search.fit(x, y)
    tuned_params = random_search.best_params_
    print tuned_params

    # Tuned Hyper parameters
    penalty_options = tuned_params['logisticregression__penalty']
    C_range = tuned_params['logisticregression__C']
    solver = tuned_params['logisticregression__solver']
    max_iter = tuned_params['logisticregression__max_iter']

    print('\nLR Best Tuned Model')
    tuned_clf = LogisticRegression(penalty=penalty_options, C=C_range, random_state=7, solver=solver, max_iter=max_iter)
    modelEvaluator(tuned_clf, x, y, folds)


def svmClassifier(x, y, folds):
    n_iter_search = 50
    skf = StratifiedKFold(folds)

    clf = SVC(kernel='linear', random_state=7)
    pipe = make_pipeline(preprocessing.StandardScaler(), clf)

    # Hyper parameters
    C = sp_rand()

    param_dist = dict(svc__C=C)

    random_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search, cv=skf,
                                       scoring='roc_auc')
    random_search.fit(x, y)
    tuned_params = random_search.best_params_
    print tuned_params

    # Tuned Hyper parameters
    C = tuned_params['svc__C']

    # print('\nSVM(Linear) Best Tuned Model')
    tuned_clf = SVC(kernel='linear', C=C, random_state=7)
    modelEvaluator(tuned_clf, x, y, folds)


def randomForest(x, y, folds):
    n_iter_search = 50
    skf = StratifiedKFold(folds)

    clf = RandomForestClassifier(n_jobs=-1, random_state=7)
    pipe = make_pipeline(preprocessing.StandardScaler(), clf)

    # Hyper parameters
    min_samples_split = sp_randint(2, 10)
    n_estimators = sp_randint(1, 300)
    max_depth = sp_randint(1, 4)

    param_dist = dict(randomforestclassifier__min_samples_split=min_samples_split,
                      randomforestclassifier__n_estimators=n_estimators, randomforestclassifier__max_depth=max_depth)

    random_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search, cv=skf,
                                       scoring='roc_auc')
    random_search.fit(x, y)
    tuned_params = random_search.best_params_
    print tuned_params

    # Tuned Hyper parameters
    min_samples_split = tuned_params['randomforestclassifier__min_samples_split']
    n_estimators = tuned_params['randomforestclassifier__n_estimators']
    max_depth = tuned_params['randomforestclassifier__max_depth']

    # print('\nSVM(Linear) Best Tuned Model')
    tuned_clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                       max_depth=max_depth, n_jobs=-1, random_state=7)
    modelEvaluator(tuned_clf, x, y, folds)


def decisionTree(x, y, folds):
    n_iter_search = 50
    skf = StratifiedKFold(folds)

    clf = tree.DecisionTreeClassifier(random_state=7)
    pipe = make_pipeline(preprocessing.StandardScaler(), clf)

    # Hyper parameters
    criterion = ['gini', 'entropy']
    max_depth = sp_randint(1, 4)
    min_samples_split = sp_randint(2, 10)
    min_samples_leaf = sp_randint(1, 10)

    param_dist = dict(decisiontreeclassifier__criterion=criterion, decisiontreeclassifier__max_depth=max_depth,
                      decisiontreeclassifier__min_samples_split=min_samples_split,
                      decisiontreeclassifier__min_samples_leaf=min_samples_leaf)

    random_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search, cv=skf,
                                       scoring='roc_auc')
    random_search.fit(x, y)
    tuned_params = random_search.best_params_
    print tuned_params

    # Tuned Hyper parameters
    criterion = tuned_params['decisiontreeclassifier__criterion']
    max_depth = tuned_params['decisiontreeclassifier__max_depth']
    min_samples_split = tuned_params['decisiontreeclassifier__min_samples_split']
    min_samples_leaf = tuned_params['decisiontreeclassifier__min_samples_leaf']

    # print('\nSVM(Linear) Best Tuned Model')
    tuned_clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                            random_state=7)
    modelEvaluator(tuned_clf, x, y, folds)


def LDA(x, y, folds):
    n_iter_search = 50
    skf = StratifiedKFold(folds)

    clf = LinearDiscriminantAnalysis()
    pipe = make_pipeline(preprocessing.StandardScaler(), clf)

    # Hyper parameters
    solver = ['svd', 'lsqr', 'eigen']

    param_dist = dict(lineardiscriminantanalysis__solver=solver)

    random_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search, cv=skf,
                                       scoring='roc_auc')
    random_search.fit(x, y)
    tuned_params = random_search.best_params_
    print tuned_params

    # Tuned Hyper parameters
    solver = tuned_params['lineardiscriminantanalysis__solver']
    # shrinkage = tuned_params['lineardiscriminantanalysis__shrinkage']

    # print('\nSVM(Linear) Best Tuned Model')
    tuned_clf = LinearDiscriminantAnalysis(solver=solver)
    modelEvaluator(tuned_clf, x, y, folds)


def KNN(x, y, folds):
    n_iter_search = 50
    skf = StratifiedKFold(folds)

    clf = KNeighborsClassifier(n_neighbors=2, algorithm='brute')
    pipe = make_pipeline(preprocessing.StandardScaler(), clf)

    # Hyper parameters
    weights = ['uniform', 'distance']
    p = [1, 2]

    param_dist = dict(kneighborsclassifier__weights=weights, kneighborsclassifier__p=p)

    random_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search, cv=skf,
                                       scoring='roc_auc')
    random_search.fit(x, y)
    tuned_params = random_search.best_params_
    print tuned_params

    # Tuned Hyper parameters
    weights = tuned_params['kneighborsclassifier__weights']
    p = tuned_params['kneighborsclassifier__p']

    # print('\nSVM(Linear) Best Tuned Model')
    tuned_clf = KNeighborsClassifier(n_neighbors=2, algorithm='brute', weights=weights, p=p)
    modelEvaluator(tuned_clf, x, y, folds)