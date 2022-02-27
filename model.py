import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn import linear_model
#import sqlite3
#from bs4 import BeautifulSoup
#import re
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
#import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
#from sklearn.calibration import CalibratedClassifierCV
#from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import pickle

if (__name__ == "__main__"):



    #############################################################################################

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    #############################################################################################

    # Dropping id from test data and dropping id and target from train data

    test_id = test_df['id']
    test_df.drop('id', axis=1, inplace=True)
    y = train_df['target']
    train = train_df.drop(['id', 'target'], axis=1)

    #############################################################################################

    # Identifying categorial, binary, calc features

    all_features = train.columns.values
    # Categorial features
    cat_features = []
    for i in train.columns.values:
        if i.endswith('cat'):
            cat_features.append(i)
        else:
            continue
    # Binary features
    bin_features = []
    for i in train.columns.values:
        if i.endswith('bin'):
            bin_features.append(i)
        else:
            continue
    # Calc features
    calc_features = []
    for i in train.columns.values:
        if i.startswith('ps_calc'):
            calc_features.append(i)
        else:
            continue
    # Calc binary features
    calc_bin_features = []
    for i in calc_features:
        if i.endswith('bin'):
            calc_bin_features.append(i)
        else:
            continue

    calc_num_features = list(set(calc_features) - set(calc_bin_features))
    num_features_with_calc = list((set(train.columns.values) - set(cat_features)) - set(bin_features))
    num_features_wo_calc = list(set(num_features_with_calc) - set(calc_features))


    # num_features_wo_calc

    #############################################################################################

    # Function for calculating gini coefficient

    def gini(actual, predicted, cmpcol=0, sortcol=1):
        # Assert statement if length of actual is equal to length of predicted
        assert (len(actual) == len(predicted))

        all = np.asarray(np.c_[actual, predicted, np.arange(len(actual))], dtype=np.float)
        all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]

        # To check total loss
        totalLoss = all[:, 0].sum()

        # Getting gini sum
        gini_sum = all[:, 0].cumsum().sum() / totalLoss
        gini_sum -= (len(actual) + 1) / 2

        # gini sum divided by len of actual gives us gini coefficient
        return gini_sum / len(actual)


    # Function for calculating the normalized gini coefficient

    def gini_normalized(a, p):
        return gini(a, p) / gini(a, a)  # calculating normalized gini coefficient


    #############################################################################################
    # Preparing the Data
    # Function to fill in the missing values of categorical features

    def filling_missing_values(data):
        for i in tqdm(data.columns.values):
            if data[data[i] == -1].shape[0] > 0:
                if i == 'ps_car_03_cat' or i == 'ps_car_05_cat':
                    continue
                # Applying mode for ps_ind_05_cat and ps_car_07_cat features
                elif i == 'ps_ind_05_cat' or i == 'ps_car_07_cat':
                    lst = data[i].apply(lambda x: data[i].mode() if x == -1 else x)
                    data[i] = pd.Series(lst)
                else:
                    # Applying mean for ps_ind_02_cat, ps_ind_04_cat, ps_car_01_cat, ps_car_02_cat, ps_car_09_cat features
                    lst = data[i].apply(lambda x: data[i].mean() if x == -1 else x)
                    data[i] = pd.Series(lst)
            else:
                continue
        return data


    #############################################################################################

    train_without_missing_values = train
    test_without_missing_values = test_df

    print("Train without missing values --> Shape :", train_without_missing_values.shape)
    print("Test without missing values --> Shape :", test_without_missing_values.shape)

    #############################################################################################

    # Adding an extra feature "missing" to the dataset which consists of all the missing values in a row.
    # train_without_missing_values['missing'] = (train == -1).sum(axis=1)
    # test_without_missing_values['missing'] = (test_df == -1).sum(axis=1)

    #############################################################################################

    print("Shape of New Train --> without any missing values and with 'missing' column :",
          train_without_missing_values.shape)
    print("Shape of New Test --> without any missing values and with 'missing' column :",
          test_without_missing_values.shape)

    #############################################################################################

    # Dropping All calc features as they are just random noise
    train_without_missing_values.drop(calc_features, axis=1, inplace=True)
    test_without_missing_values.drop(calc_features, axis=1, inplace=True)

    #############################################################################################

    print("Train Data ready for modelling --> Shape :", train_without_missing_values.shape)
    print("Test Data ready for modelling --> Shape :", test_without_missing_values.shape)

    print(
        "=================================================================================================================")
    print(train_without_missing_values.columns.values)
    print(
        "=================================================================================================================")
    print(test_without_missing_values.columns.values)
    print(
        "=================================================================================================================")


    #############################################################################################

    # PREPARING THE MODEL

    # Function for one-hot-encoding categorical features
    def one_hot_encoding(train, test, cat_features):
        temp = pd.concat([train, test])
        temp = pd.get_dummies(temp, columns=cat_features)
        train = temp.iloc[:train.shape[0], :]
        test = temp.iloc[train.shape[0]:, :]
        return train, test


    #############################################################################################

    # For Actual submission
    train_ohe, test_ohe = one_hot_encoding(train_without_missing_values, test_without_missing_values, cat_features)

    #############################################################################################

    # splitting the train dataset for cross validation and one_hot_encoding categorical features
    X_train, X_cv, y_train, y_cv = train_test_split(train_without_missing_values, y, test_size=0.2, stratify=y,
                                                    random_state=2019)
    X_train, X_cv = one_hot_encoding(X_train, X_cv, cat_features)

    #############################################################################################

    # Using skLearn StandardScalar to scale the values
    scaler = StandardScaler()
    scaler.fit(X_train[num_features_wo_calc])

    X_train[num_features_wo_calc] = scaler.transform(X_train[num_features_wo_calc])
    X_cv[num_features_wo_calc] = scaler.transform(X_cv[num_features_wo_calc])

    #############################################################################################

    # Checking the shape of train and cross validation data
    print("X_train Shape : ", X_train.shape)
    print("X_cv Shape : ", X_cv.shape)

    #############################################################################################
    # LIGHT GBM
    # Final Parameters of the model
    lgb_params = {}
    lgb_params['n_estimators'] = 1300
    lgb_params['learning_rate'] = 0.01
    lgb_params['num_leaves'] = 30
    lgb_params['feature_fraction'] = 0.8
    lgb_params['min_data_in_leaf'] = 1400
    lgb_params['lambda_l1'] = 1
    lgb_params['lambda_l2'] = 1
    lgb_params['bagging_freq'] = 1
    lgb_params['bagging_fraction'] = 0.8

    #############################################################################################

    lgb_model = LGBMClassifier(**lgb_params)
    #lgb_model.fit(X_train.values, y_train.values)
    #lgb_pred = lgb_model.predict_proba(X_cv)[:, 1]

    #############################################################################################
    #roc_auc_score(y_cv, lgb_pred), gini_normalized(y_cv, lgb_pred)
    #############################################################################################
    print("Number of train_ohe", train_ohe.columns)
    lgb_model.fit(train_ohe.values, y.values)
    lgb_test_pred = lgb_model.predict_proba(test_ohe.values)[:, 1]


    submission = pd.DataFrame()
    submission['id'] = test_id
    submission['target'] = lgb_test_pred
    submission.to_csv('final_predictions_lgb.csv', index=False)


    #############################################################################################
    # FEATURE SELECTION
    #############################################################################################
    features = train_ohe.columns.values
    #############################################################################################

    def feature_selection(fi, features, threshold, graph=True):
        # Function for feature selection.
        idx_fi_desc = np.argsort(fi)[::-1]
        features_desc = [features[i] for i in idx_fi_desc]
        fi_desc = fi[idx_fi_desc]
        # Features that are having importance, greater than the threshold are selected
        selected_features = [features_desc[i] for i in range(len(fi_desc)) if fi_desc[i] > threshold]
        selected_fi = [fi_desc[i] for i in range(len(fi_desc)) if fi_desc[i] > threshold]

        # Plotting the graph
        if graph:
            plt.figure(figsize=(10, 30))
            plt.title("Feature Importance")
            plt.barh(features_desc[:len(selected_features)], fi_desc[:len(selected_features)], color='Red')

        # returning the selected features
        return selected_features



    #############################################################################################


    #############################################################################################


    selected_features_lgb = feature_selection(lgb_model.feature_importances_, features, 0)

    print("Total number of features selected by LightGBM :", len(selected_features_lgb))
    print("Features selected by LightGBM :", selected_features_lgb)

    os.chdir("static/files")
    # save the model to disk
    filename = 'model.pkl'
    pickle.dump(lgb_model, open(filename, 'wb'))

