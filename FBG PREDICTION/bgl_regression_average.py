
import numpy as np
from numpy.random import seed
import random
#seed(7)
import pandas as pd
from scipy import stats
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import f_classif, f_regression, mutual_info_regression

def load_data(FILE):
    # load the data
    #random.seed(7)
    df = pd.read_csv(FILE, sep=',', header = 0)
    print('Number of test subjects avaibale: ',len(df['id'].value_counts()))
    print('Number of extracted signal segments: ', df.shape[0])
    #print(df.dtypes)
    subjects = df.id.unique()
    print(subjects)

    #get_classwise_ANOVA(df)

    df['select'] = 0;
    # loop through subjects and find best.
    ZSCORE_THRESHOLD = 2
    first_count = 0;
    for subject in subjects:
        filtered_df = df.loc[df['id'] == subject]
        unique_loops = filtered_df.Loop.unique()
        #print(unique_loops)
        df1 = filtered_df[(np.abs(stats.zscore(filtered_df[['apg1', 'apg2', 'apg3', 'apg4', 'apg5', 'apg6', 'apg7', 'apg8',\
             'apgRR', 'SysAmp','TotArea', 'AreaRatio', 'PI', 'RT','PI_Sys', 'AI', 'adj_AI']])) < ZSCORE_THRESHOLD).all(axis=1)]

        # obtain the avergae across the segemnts.
        if df1.shape[0] > 1:
            mean_df = pd.DataFrame(df1.mean())
            mean_df = mean_df.transpose()
        else:
            mean_df = df1

        #aa_std = df1['apgRR'].std()
        #pi_std = df1['PI'].std()
        #mean_df['apgRR_std'] = aa_std
        #mean_df['PI_std'] = pi_std
        #rt = df1['RT'].std()
        #pi_sys = df1['PI_Sys'].std()
        #mean_df['RT_std'] = rt
        #mean_df['PI_Sys_std'] = pi_sys

        #print(mean_df)
        if first_count == 0:
            final_avg = mean_df
        else:
            final_avg = final_avg.append(mean_df)
        first_count = first_count + 1;

        #select a segment randomly
        random_subject = random.randint(0, len(unique_loops)-1)
        #print('Analyzing Subject: ', subject, '==>', len(unique_loops))
        #print(unique_loops[random_subject])
        df.loc[(df['id'] == subject) & (df['Loop'] == unique_loops[random_subject]), 'select'] = 1

    final_df = df.loc[df['select'] == 1]
    #print(final_df)
    #return final_df

    print(final_avg)
    return final_avg

def loo_regression(standardized_X, Y,TYPE):
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
    reg5 = xgboost.XGBRegressor(max_depth=3, reg_lambda=1, reg_alpha=1)
    print(reg5)
    reg7 = RandomForestRegressor(n_estimators=10, max_depth=3)  # , max_features=10
    print(reg7)
    # reg8 = SVR(kernel='poly')

    regs = [reg1, reg2, reg3, reg4, reg5, reg7]
    classifier_name = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'XGB','RF']
    table = []

    for i in range(0, len(classifier_name)):
        #print('\n')
        #print('Classifier: ', classifier_name[i])
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
        filename3 = 'output_corrupt/' +TYPE +'/Val_Result_' + classifier_name[i] + '.csv'
        temp.to_csv(filename3,header=False, index=False)

    table = pd.DataFrame(table)
    filename4 = 'output_corrupt/' + TYPE + '/result_summary.csv'
    table.to_csv(filename4,header=False, index=False)
    print(table)

def anova_test(x,y):
    print("ANOVA TEST")
    anova = f_classif(x,y)
    result = pd.DataFrame(list(anova))
    head = list(x.columns)
    result.columns = head
    result.index = ['F','P'] # Row 1 - F values, Row 2 - P values.
    print(result)
    selected_features = []
    for column in result:
        p = result.loc['P',column]
        if p < 0.01:
            selected_features.append(column)
    print("\nSelecte features by ANOVA test: ")
    print(selected_features)
    return selected_features

def get_classwise_ANOVA(final_df):
    all_features = ['Age', 'apg1', 'apg2', 'apg3', 'apg4', 'apg5', 'apg6', 'apg7', 'apg8', 'apgRR', 'SysAmp', 'TotArea', \
                    'AreaRatio', 'PI', 'RT', 'PI_Sys', 'AI', 'adj_AI']
    final_df.loc[final_df['BGL'] <= 100.0, 'Class'] = 0
    final_df.loc[(final_df['BGL'] > 100.0) & (final_df['BGL'] <= 125.0), 'Class'] = 1
    final_df.loc[final_df['BGL'] > 125.0, 'Class'] = 2

    final_df.to_csv('stat_difference.csv', header=True, index=False)
    print(final_df['Class'].value_counts())

    #multiclass
    print("Multiclass")
    anova_test(final_df[all_features], final_df[['Class']])

    #healthy vs prediabetes
    print("Healthy Vs Prediabetes")
    case1 = final_df.drop(final_df[final_df.Class == 2].index)
    case1_x = case1[all_features]
    case1_Y = case1[['Class']]
    anova_test(case1_x, case1_Y)

    #healthy vs diabetes
    print("Healthy Vs Diabetes")
    case2 = final_df.drop(final_df[final_df.Class == 1].index)
    case2_x = case2[all_features]
    case2_Y = case2[['Class']]
    anova_test(case2_x, case2_Y)

    #prediabetes vs diabetes
    print("Prediabetes Vs Diabetes")
    case4 = final_df.drop(final_df[final_df.Class == 0].index)
    case4_x = case4[all_features]
    case4_Y = case4[['Class']]
    anova_test(case4_x, case4_Y)

    # healthy vs prediabetes + diabetes
    print("Healthy vs Prediabetes + Diabetes")
    final_df.loc[final_df['Class'] == 2, 'Class'] = 1
    case3 = final_df
    case3_x = case3[all_features]
    case3_Y = case3[['Class']]
    anova_test(case3_x, case3_Y)
    #print(case3_x)
    #print(case3_Y)

def main():
    #FILE = 'input/new_full_bglFeatures_cleaned.csv'
    #FILE = 'input/corrupt_DV_bglfeatures.csv'
    FILE = 'input/noise_model_finalcor.csv'

    final_df = load_data(FILE)
    print(final_df)

    all_features = ['Age', 'apg1', 'apg2', 'apg3', 'apg4', 'apg5', 'apg6', 'apg7', 'apg8', 'apgRR', 'SysAmp', 'TotArea', \
                    'AreaRatio', 'PI', 'RT', 'PI_Sys', 'AI', 'adj_AI'] #'apgRR_std','PI_std','RT_std','PI_Sys_std'
    Y = final_df[['BGL']]
    X = final_df[all_features]

    # feature selection for regression
    # f_regres = f_regression(X,Y)
    # f_regres = mutual_info_regression(X,Y)
    # result = pd.DataFrame(list(f_regres))
    # print(result)

    # get_classwise_ANOVA(final_df)
    # final_df.to_csv('selected_features.csv', header=True, index=False)
    # quit()

    # rest the indices a
    X.reset_index(inplace=True, drop=True)
    X = pd.DataFrame(X)
    Y.reset_index(inplace=True, drop=True)
    Y = pd.DataFrame(Y)

    # Task 1 (FULL)
    print('\nTest with all features')
    loo_regression(X, Y,'FULL')

    # Feature Reduction
    scaler = StandardScaler()
    standardized_X = scaler.fit_transform(X)
    standardized_X = pd.DataFrame(standardized_X)

    # PCA
    print('\nTest with feature reduction using PCA')
    pca = PCA(0.95)
    pca.fit(standardized_X)
    fitted_X_PCA = pca.transform(standardized_X)
    fitted_X_PCA = pd.DataFrame(fitted_X_PCA)
    print('Number of features selected by PCA', fitted_X_PCA.shape[1])
    loo_regression(fitted_X_PCA, Y, 'PCA')

    # SVD
    print('\nTest with feature reduction using Truncated SVD')
    svd = TruncatedSVD(n_components=10, n_iter=7)
    svd.fit(standardized_X)
    fitted_X_SVD = svd.transform(standardized_X)
    fitted_X_SVD = pd.DataFrame(fitted_X_SVD)
    print('Number of features selected by SVD', fitted_X_SVD.shape[1])
    loo_regression(fitted_X_SVD, Y, 'SVD')

if __name__ == '__main__':
    main()