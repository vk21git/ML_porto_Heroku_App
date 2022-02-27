from flask import Flask, render_template, request, redirect, url_for, flash,jsonify
import os
from os.path import join, dirname, realpath
import pickle
import pandas as pd
#from sklearn.externals import joblib
import joblib
#from model import *
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


app = Flask(__name__)
#model = pickle.load(open('SG_finalized_model.pkl', 'rb'))

# enable debugging mode
app.config["DEBUG"] = True
app.config["CACHE_TYPE"] = "null"

# Upload folder
#UPLOAD_FOLDER = 'static/files'
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

# Root URL
@app.route('/')
def index():
     # Set The upload HTML template '\templates\index.html'
    return render_template('index.html')


# Get the uploaded files
#@app.route("/", methods=['POST'])
def uploadfiles():
      # get the uploaded file
      uploaded_file = request.files['file']
      if uploaded_file.filename != '':
           file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
          # set the file path
           uploaded_file.save(file_path)
          # save the file
      return redirect(url_for('index'))

#@app.route("/", methods=['post'])
@app.route('/',methods=['POST'])
def predict():

    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)


    os.chdir("static/files")
    filename = 'model.pkl'
    RFE_columns = pd.read_csv('RFE_features_1.csv').columns
    #load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))

    #print(filename)
    #print(loaded_model)

    # Load the test set
    df_test = pd.read_csv('sample.csv')

    RFE_columns = [col for col in RFE_columns if col not in 'missing']  # Test wont have the label column
    df_test = df_test[RFE_columns]

    def one_hot_encoding(train, test, cat_features):
        temp = pd.concat([train, test])
        temp = pd.get_dummies(temp, columns=cat_features)
        train = temp.iloc[:train.shape[0], :]
        test = temp.iloc[train.shape[0]:, :]
        return train, test

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    #############################################################################################

    # Dropping id from test data and dropping id and target from train data

    test_id = test_df['id']
    test_df.drop('id', axis=1, inplace=True)
    y = train_df['target']
    train = train_df.drop(['id', 'target'], axis=1)

    train_without_missing_values = train
    test_without_missing_values = test_df

    # Categorial features
    cat_features = []
    for i in train.columns.values:
        if i.endswith('cat'):
            cat_features.append(i)
        else:
            continue

    # Calc features
    calc_features = []
    for i in train.columns.values:
        if i.startswith('ps_calc'):
            calc_features.append(i)
        else:
            continue

    # Dropping All calc features as they are just random noise
    train_without_missing_values.drop(calc_features, axis=1, inplace=True)
    test_without_missing_values.drop(calc_features, axis=1, inplace=True)

    #############################################################################################
    # splitting the train dataset for cross validation and one_hot_encoding categorical features
    X_train, X_cv, y_train, y_cv = train_test_split(train_without_missing_values, y, test_size=0.2, stratify=y,
                                                    random_state=2019)
    X_train, X_cv = one_hot_encoding(X_train, X_cv, cat_features)

    #lgb_model = LGBMClassifier(**lgb_params)
    #lgb_model.fit(X_train.values, y_train.values)

    RFE_columns = pd.read_csv('RFE_features_1.csv').columns
    #load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))

    # Load the test set
    df_test = pd.read_csv('sample.csv')
    RFE_columns = [col for col in RFE_columns if col not in 'missing']  # Test wont have the label column
    #RFE_columns = [col for col in RFE_columns]  # Test wont have the label column
    df_test = df_test[RFE_columns]

    #print("Number of df_test", df_test.columns)

    # For Actual submission
    train_ohe_1, test_ohe_1 = one_hot_encoding(train_without_missing_values, df_test, cat_features)

    print("Number of test_ohe_1", test_ohe_1)

    lgb_test_pred = loaded_model.predict_proba(test_ohe_1.values)[:, 1]

    print("lgb_test_pred = ", lgb_test_pred)


    if lgb_test_pred>0.25:
        prediction = "Driver will initiate an auto insurance claim in the next year"
    else:
        prediction = "Driver will NOT initiate an auto insurance claim in the next year"

    #return render_template('index.html', prediction_text='Result for this Record is = {}'.format(prediction))

    return jsonify({'Prediction Result': prediction})
    #return redirect(url_for('index'))

if (__name__ == "__main__"):
     app.run(debug=True)