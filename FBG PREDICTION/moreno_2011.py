import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import numpy as np
from numpy.random import seed
seed(7)
from sklearn.metrics import mean_absolute_error
import xgboost
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

#df = pd.read_csv('input/moreno_input/corrupt_moreno_features.csv', sep=',', header = None)
#df = pd.read_csv('input/moreno_input/corrupt_DV_moreno.csv', sep=',', header = None)
#df = pd.read_csv('input/moreno_input/noise_model_final_moreno.csv', sep=',', header = None)
df = pd.read_csv('input/moreno_input/noise_model_finalcor_moreno.csv', sep=',', header = None)
Y = df[0]
X = df.drop(0,1)
#print(X)


def loo_regression(standardized_X, Y):
    'Leave one out regression and provide the mean absolute error for train test'

    # convert to numpy arrays
    Y = np.array(Y)
    standardized_X = np.array(standardized_X)

    loocv = model_selection.LeaveOneOut()

    reg1 = LinearRegression()
    reg2 = Ridge()
    print(reg2)
    reg3 = Lasso()
    print(reg3)
    reg4 = ElasticNet()
    print(reg4)
    reg5 = xgboost.XGBRegressor(max_depth=3,reg_lambda=1,reg_alpha=1)
    print(reg5)
    reg7 = RandomForestRegressor(n_estimators=10,max_depth=3) #, max_features=10
    print(reg7)
    #reg8 = SVR(kernel='poly')

    regs = [reg1, reg2, reg3, reg4, reg5, reg7]
    classifier_name = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'XGB','RF']
    table = []

    for i in range(0, len(classifier_name)):
        print('\n')
        print('Classifier: ', classifier_name[i])
        reg = regs[i]
        loop = 0
        temp = []

        temp_summary = []
        temp_train_err = []
        temp_test_err = []

        for train_index, test_index in loocv.split(standardized_X):
            loop = loop + 1
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = standardized_X[train_index], standardized_X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            # train_accuracy.append(reg.score(X_train, y_train))

            reg.fit(X_train, y_train)

            y_train_predict = reg.predict(X_train)
            train_error = mean_absolute_error(y_train, y_train_predict)

            y_test_predict = reg.predict(X_test)
            test_error = mean_absolute_error(y_test, y_test_predict)
            temp.append([ y_test[0],round(y_test_predict[0],2) ])

            temp_train_err.append(train_error)
            temp_test_err.append(test_error)

        train_mean = round(np.mean(temp_train_err),2)
        train_std  = round(np.std(temp_train_err),2)
        test_mean  = round(np.mean(temp_test_err),2)
        test_std   = round(np.std(temp_test_err),2)
        temp_summary.extend([classifier_name[i],train_mean, train_std, test_mean, test_std])
        table.append(temp_summary)

        temp = pd.DataFrame(temp)
        filename3 = 'output/moreno_corrupt/Val_Result_' + classifier_name[i] + '.csv'
        temp.to_csv(filename3,header=False, index=False)

    table = pd.DataFrame(table)
    table.to_csv('output/moreno_corrupt/result_summary.csv',header=False, index=False)
    print(table)


print('\nTest with all features')
loo_regression(X, Y)

