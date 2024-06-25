import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from joblib import load
from joblib import dump

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def replace_values(y):
    return y.replace({'B': 0, 'D': 4, 'M': 1, 'NK': 2, 'T': 3}, regex=True)

def return_values(y):
    return y.replace({0: 'B', 4: 'D', 1: 'M', 2: 'NK', 3: 'T'}, regex=True)

def model_info(y_real, y_pred, title, labels=['B', 'D', 'M', 'NK', 'T']):
    cm = confusion_matrix(y_real, y_pred)
    class_sums = cm.sum(axis=1)
    normalized_conf_matrix = cm / class_sums[:, np.newaxis]
    sns.heatmap(normalized_conf_matrix, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels, fmt='.2f')
    plt.xlabel('Predvidjene vrednosti')
    plt.ylabel('Stvarne vrednosti')
    plt.title(title)
    plt.show()

    print('Matrica konfuzije:\n', cm)
    print('Accuracy score: ', accuracy_score(y_real, y_pred))
    print('Precision score: ', precision_score(y_real, y_pred, average='weighted', zero_division=0))
    print('Recall score: ', recall_score(y_real, y_pred, average='weighted'))
    print('F1 score: ', f1_score(y_real, y_pred, average='weighted'))

def build_model(model, X_train, X_test, y_train, y_test, title, classes=['B', 'D', 'M', 'NK', 'T']):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    model_info(y_test, y_pred, title, classes)

    return model

def load_test(data, model, title, classes=['B', 'D', 'M', 'NK', 'T']):
    y_pred = model.predict(data.drop(['Group'], axis=1))
    model_info(data['Group'], y_pred, title, labels=classes)
    return y_pred

def test_on_self(data, params, algorithm, title, classes=['B', 'D', 'M', 'NK', 'T']):
    model = algorithm(**params, random_state=55)
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['Group'], axis=1), data['Group'], test_size=0.3, random_state=55)
    model = build_model(model, X_train, X_test, y_train, y_test, title, classes=classes)
    return model.predict(data.drop(['Group'], axis=1))

def test(data, algorithm, model, params, title, classes=['B', 'D', 'M', 'NK', 'T']):
    if(algorithm == XGBClassifier):
        data['Group'] = replace_values(data['Group'])
    results = pd.DataFrame()
    results[title + '_self'] = test_on_self(data, params, algorithm, title + '_self', classes)
    results[title] = load_test(data, model, title)
    if(algorithm == XGBClassifier):
        results[title + '_self'] = return_values(results[title + '_self'])
        results[title] = return_values(results[title])
    return results

def write_results(filename, classes):
    datapath = '../data/preprocessed_data_' + filename + '.csv'
    data = pd.read_csv(datapath)
    results = pd.DataFrame()

    rf = load('../models/trained_models/rf.joblib')
    rf_over = load('../models/trained_models/rf_over.joblib')

    rf_params = {'max_depth': 15,
    'min_samples_leaf': 1,
    'min_samples_split': 5,
    'n_estimators': 300}

    results = pd.concat([results ,test(data, RandomForestClassifier, rf, rf_params, 'rf', classes)], axis=1)
    results = pd.concat([results ,test(data, RandomForestClassifier, rf_over, rf_params, 'rf_over', classes)], axis=1)

    log_reg = load('../models/trained_models/log_reg.joblib')
    log_reg_cw = load('../models/trained_models/log_reg_cw.joblib')
    log_reg_over = load('../models/trained_models/log_reg_over.joblib')

    log_reg_params = {'C': 0.001, 'penalty': 'l2', 'solver': 'sag'}
    log_reg_cw_params = {'C': 0.001, 'penalty': 'l2', 'solver': 'sag', 'class_weight' : 'balanced'}

    results = pd.concat([results ,test(data, LogisticRegression, log_reg, log_reg_params,'log_reg', classes)], axis=1)
    results = pd.concat([results ,test(data, LogisticRegression, log_reg_cw, log_reg_cw_params,'log_reg_cw', classes)], axis=1)
    results = pd.concat([results ,test(data, LogisticRegression, log_reg_over, log_reg_params,'log_reg_over', classes)], axis=1)

    nn = load('../models/trained_models/nn.joblib')
    nn_over = load('../models/trained_models/nn_over.joblib')

    nn_params = {'alpha': 1e-05,
    'batch_size': 32,
    'hidden_layer_sizes': (50,),
    'learning_rate_init': 0.001,
    'max_iter': 1000}

    results = pd.concat([results ,test(data, MLPClassifier, nn, nn_params,'nn', classes)], axis=1)
    results = pd.concat([results ,test(data, MLPClassifier, nn_over, nn_params,'nn_over', classes)], axis=1)

    xg = load('../models/trained_models/xg.joblib')
    xg_over = load('../models/trained_models/xg_over.joblib')

    xg_params = {'learning_rate': 0.2, 'max_depth': 4, 'n_estimators': 300}

    results = pd.concat([results ,test(data, XGBClassifier, xg, xg_params,'xg')], axis=1)
    results = pd.concat([results ,test(data, XGBClassifier, xg_over, xg_params,'xg_over')], axis=1)

    return results