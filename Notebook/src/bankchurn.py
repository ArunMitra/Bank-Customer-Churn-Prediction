
import matplotlib.pyplot as plt
plt.style.use('ggplot') #'fivethirtyeight')
import seaborn as sns

import numpy as np
import pandas as pd

import scipy.stats as scs
import scipy

import itertools
from collections import Counter

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def prepare_and_split_data(file_name_with_path):
    ''' Prepares data and splits it
        Parameters:
            file_name_with_path : string

        Returns:
            Full_X       : 2D pd dataframe containing the full cleaned data features
            X_train      : 2D pd dataframe containing training data features
            X_final_test : 2D pd dataframe containing test data features
            Full_y       : 1D np array conatining the full label data
            y_train      : 1D np array containing training labels
            y_final_test : 1D np array containing test labels
    '''
    # read in the data
    df = pd.read_csv('../data/bank_churn.csv', ';')

    # drop the 'customer_id', as it carries no signal
    df = df.drop(['customer_id'], axis=1)

    # change 'gender' to 0 for female and 1 for male
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

    # Encode the category 'country'
    X = pd.get_dummies(df)

    # Take the label out of X and into y, take out the redundant feature 'country_Spain' from X
    Full_y = X['churn']
    Full_X = X.drop(['country_Spain', 'churn'], axis=1)

    # Split the data
    X_train, X_final_test, y_train, y_final_test = train_test_split(Full_X, Full_y, test_size=0.10, train_size=0.90, random_state=13)
    return Full_X, X_train, X_final_test, Full_y, y_train, y_final_test

#### Helper function: run_models( )

def run_models(estimator_class, X_train, y_train, n_splits=5, stratified=False, sampling=None):

    ''' Runs a list of models and returns scores
        Parameters:
            estimator_class : a list of klearn.linear_model or sklearn.ensemble classes
            X_train         : 2D pd dataframe with feature data
            y_train         : 1D np array with label data
            n_splits        : the number of KFolds to iterate
            stratified      : whether to use stratified KFolds
            sampling        :
                              None          : do not under- or over- sample
                              'SMOTE'       : use SMOTE oversampling
                              'UNDER'       : undersample majority class
                              'OVER'        : oversample minority class
                              'SMOTE&UNDER' : use a combination of SMOTE & undersampling


        Returns:
            - A list of tuples (one for each model cross-validated)
              - each tuple contains 3 lists for accuracy, precision and recall scores
                - each scores list (in each tuple) contains n_splits scores (one for each KFold split)
    '''
    if stratified:
        kf = StratifiedKFold(n_splits, shuffle=True)
    else:
        kf = KFold(n_splits, shuffle=True)

    accuracy = []
    precision = []
    recall = []
    for train_idx, test_idx in kf.split(X_train, y_train):

        # Setup the fold's test data
        X_test_fold = X_train.iloc[test_idx]
        y_test_fold = y_train.iloc[test_idx]

        # Now setup the fold's train data as per the kf.split
        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]


        # Then balance the classes as per the sampling parameter

        if sampling == 'SMOTE':

            # oversample the fold's train data to balance the classes
            oversample = SMOTE()
            X_train_fold, y_train_fold \
                = oversample.fit_resample(X_train_fold, y_train_fold)

        elif sampling == 'UNDER':

            # First put X_train and y_train together
            Xy_temp = pd.concat([X_train_fold, y_train_fold], axis=1)
            # Take all samples that are in the minority class
            Xy_temp_minority = Xy_temp[Xy_temp['churn'] == 1]
            # Take an equal number of samples from the majority class
            Xy_temp_majority = Xy_temp[Xy_temp['churn'] == 0] \
                                .sample(Xy_temp_minority.shape[0])
            # then append
            Xy_temp_balanced = Xy_temp_minority.append(Xy_temp_majority)

            # Now pull out y_train_fold data
            y_train_fold = Xy_temp_balanced['churn']
            # and drop the label to get the X_train_fold data
            X_train_fold = Xy_temp_balanced.drop(['churn'], axis=1)

        elif sampling == 'OVER':

            # First put X_train and y_train together
            Xy_temp = pd.concat([X_train_fold, y_train_fold], axis=1)
            # Take all samples that are in the majority class
            Xy_temp_majority = Xy_temp[Xy_temp['churn'] == 0]
            # Take an equal number of samples from the minority class
            Xy_temp_minority = Xy_temp[Xy_temp['churn'] == 1] \
                                .sample(Xy_temp_majority.shape[0], replace=True)
            # then append
            Xy_temp_balanced = Xy_temp_majority.append(Xy_temp_minority)

            # Now pull out y_train_fold data
            y_train_fold = Xy_temp_balanced['churn']
            # and drop the label to get the X_train_fold data
            X_train_fold = Xy_temp_balanced.drop(['churn'], axis=1)

        elif sampling == 'SMOTE&UNDER':

            # setup
            oversample = SMOTE(sampling_strategy=0.5)
            undersample = RandomUnderSampler(sampling_strategy=1)
            # oversample majority class
            X_train_fold, y_train_fold \
                = oversample.fit_resample(X_train_fold, y_train_fold)
            # then undersample minority
            X_train_fold, y_train_fold \
                = undersample.fit_resample(X_train_fold, y_train_fold)

        # fit model, predict, compute scores
        estimator = estimator_class()
        estimator.fit(X_train_fold, y_train_fold)
        y_preds_fold = estimator.predict(X_test_fold)
        accuracy.append(accuracy_score(y_test_fold, y_preds_fold))
        precision.append(precision_score(y_test_fold, y_preds_fold))
        recall.append(recall_score(y_test_fold, y_preds_fold))

    return (accuracy, precision, recall)

#### Helper function: show_model_scores( )

def show_model_scores(estimators, results, n_splits=5, suffix=''):

    ''' Plots and prints a list of model scores after cross validation
        Parameters:
            estimators  : a list of sklearn.linear_model or sklearn.ensemble model instances
            results     : a list of tuples (each containing 3 lists for Accuracy, Precision, and
                          Recall scores for each KFold iteration) for each estimator
            n_splits    : the number of KFolds that were used in the cross_validation
            suffix      : a string containing information about sampling and stratification to be appended
                        to the model name and used in the plot title, and score print out

        Returns:
            None (directly plots and prints)
    '''
    fig, axs = plt.subplots(1, len(results), figsize=(15, 5))

    for i, estimator in enumerate(estimators):

        axs[i].set_title(estimator.__name__ + suffix, fontsize=10)
        axs[i].plot(range(1, (n_splits+1)), results[i][0], label='Accuracy', c='green')
        axs[i].plot(range(1, (n_splits+1)), results[i][1], label='Precision', c='red')
        axs[i].plot(range(1, (n_splits+1)), results[i][2], label='Recall', c='blue')
        axs[i].set_ylabel('Accuracy, Precision, Recall', fontsize=10)
        axs[i].set_xlabel('Number of folds', fontsize=10)
        axs[i].legend()

        print('\n***********************************************')
        print(estimator.__name__ + suffix)
        print('***********************************************')
        print('Average Accuracy  = ', round(np.mean(results[i][0]), 2))
        print('Average Precision = ', round(np.mean(results[i][1]), 2))
        print('Average Recall    = ', round(np.mean(results[i][2]), 2))
        print('***********************************************')

    fig.savefig('../Images/ModelScores'+suffix)

#### Helper function: gridsearch_with_output( )

def gridsearch_with_output(estimator, parameter_grid, X_train, y_train):
    ''' Performs gridsearch and returns the best parameters and model
        Parameters:
            estimator     : the type of model (e.g. RandomForestRegressor())
            paramter_grid : dictionary defining the gridsearch parameters
            X_train       : 2d pd DataFrame
            y_train       : 1d numpy array
        Returns:
            best parameters and model fit with those parameters
    '''
    model_gridsearch = GridSearchCV(estimator,
                                    parameter_grid,
                                    n_jobs=-1,
                                    verbose=True,
                                    scoring='recall')
    model_gridsearch.fit(X_train, y_train)
    best_params = model_gridsearch.best_params_
    model_best = model_gridsearch.best_estimator_
    print("\nResult of gridsearch:")
    print("{0:<20s} | {1:<8s} | {2}".format("Parameter", "Optimal", "Gridsearch values"))
    print("-" * 55)
    for param, vals in parameter_grid.items():
        print("{0:<20s} | {1:<8s} | {2}".format(str(param),
                                                str(best_params[param]),
                                                str(vals)))
    return best_params, model_best

#### Helper function: display_default_and_gsearch_model_results( )

def display_default_and_gsearch_model_results(model_default, model_best,
                                              X_test, y_test):
    ''' Displays gridsearch results
        Parameters:
            model_default : fit model using initial parameters
            model_best    : fit model using parameters from gridsearch
            X_test        : 2d numpy array
            y_test        : 1d numpy array
        Return:
            None, but prints out recall the default and model with gridsearched parameters
    '''
    name = model_default.__class__.__name__ # for printing
    y_test_pred = model_best.predict(X_test)
    recall = recall_score(y_test, y_test_pred)
    print("Results for {0}".format(name))
    print("Gridsearched model Recall: {:0.3f}".format(recall))
    y_test_pred = model_default.predict(X_test)
    recall = recall_score(y_test, y_test_pred)
    print("     Default model Recall: {:0.3f}".format(recall))

#### Helper function: run_models_on_final_test( )

def run_models_on_final_test(model1, model2, X_train, y_train, X_final_test, y_final_test):

    ''' Runs 2 best models on final test data and shows results
        Parameters:
            model1 : 1 of 2 final models
            model2 : 2 of 2 final models
            X_test : 2d pd Dataframe with training features data
            y_test : 1d numpy array with training label data
            X_final_test : 2d pd Dataframe with final test features data
            y_final_test : 1d numpy array with final test label data
        Return:
            model  : The model that is better for Recall
            X_train_balanced : 2D pd DataFrame with oversampled training feature data
            y_train_balanced : 1D np array with oversampled training label data
            (also prints out scores for the models)
    '''
    # Setup the data

    # First put X_train and y_train together
    Xy_temp = pd.concat([X_train, y_train], axis=1)
    # Take all samples that are in the majority class
    Xy_temp_majority = Xy_temp[Xy_temp['churn'] == 0]
    # Take an equal number of samples from the minority class
    Xy_temp_minority = Xy_temp[Xy_temp['churn'] == 1] \
                        .sample(Xy_temp_majority.shape[0], replace=True)
    # then append
    Xy_temp_balanced = Xy_temp_majority.append(Xy_temp_minority)

    # Now pull out y_train_fold data
    y_train_balanced = Xy_temp_balanced['churn']
    # and drop the label to get the X_train_fold data
    X_train_balanced = Xy_temp_balanced.drop(['churn'], axis=1)# data prep

    # Run the models

    # RandomForestClassifier
    # model1 = RandomForestClassifier()
    model1.fit(X_train_balanced, y_train_balanced)
    y_pred = model1.predict(X_final_test)
    print('\n********************************************')
    print('Model: ' + model1.__class__.__name__)
    print('********************************************')
    print('Accuracy  = ', round(accuracy_score(y_final_test, y_pred), 2))
    print('Precision = ', round(precision_score(y_final_test, y_pred), 2))
    print('Recall    = ', round(recall_score(y_final_test, y_pred), 2))
    print('********************************************')

    model1_recall = recall_score(y_final_test, y_pred)

    # GradientBoostingClassifier
    # model2 = GradientBoostingClassifier()
    model2.fit(X_train_balanced, y_train_balanced)
    y_pred = model2.predict(X_final_test)
    print('\n********************************************')
    print(f'Model: {model2.__class__.__name__}')
    print('********************************************')
    print('Accuracy  = ', round(accuracy_score(y_final_test, y_pred), 2))
    print('Precision = ', round(precision_score(y_final_test, y_pred), 2))
    print('Recall    = ', round(recall_score(y_final_test, y_pred), 2))
    print('********************************************')

    model2_recall = recall_score(y_final_test, y_pred)

    if model2_recall >= model1_recall:
        print('********************************************')
        print('Best model for Recall: ', model2.__class__.__name__)
        print('********************************************')
        return model2, X_train_balanced, y_train_balanced
    else:
        print('********************************************')
        print('Best model for Recall: ', model1.__class.__name__)
        print('********************************************')
        return model1, X_train_balanced, y_train_balanced

#### Helper function: plot_profit_curve( )

def plot_profit_curve(best_model, X_final_test, y_final_test, incentive_cost=200, incr_revenue=1000):

    ''' Plots the profit curve
        Parameters:
            best_model: the chosen model
            X_final_test   : 2d numpy array of the final test features data
            y_final_test   : 1d numpy array of the final test labels data
            incentive_cost : numeric : the incentive cost in dollars (default=$200)
            incr_revenue   : numeris : the revenue from a retained cutomer in dollars (default=$1,000)
        Return:
            None, but plots the profit curve
    '''

    # First, work out the probability threshholds and profits
    y_probas = best_model.predict_proba(X_final_test)
    thresholds = np.linspace(0, 1, 100)

    incr_profit_TN = 0
    incr_profit_FP = -1*(incentive_cost)
    incr_profit_FN = -1*(incr_revenue)
    incr_profit_TP = (incr_revenue - incentive_cost) # assuming retention program is 100% effective

    cost_benefit_matrix = np.array([[incr_profit_TP, incr_profit_FP],
                                    [incr_profit_FN, incr_profit_TN]])
    profits = []
    for thresh in thresholds:
        classes = y_probas[:, 1] > thresh
        classes = [1 if c == True else 0 for c in classes]

        [[tn, fp], [fn, tp]] = confusion_matrix(y_final_test, classes)
        conf_matrix = np.array([[tp, fp], [fn, tn]])

        profit = np.sum(conf_matrix * cost_benefit_matrix)
        profits.append(profit)

    # Now plot the profit curve
    max_prof = np.argmax(profits)
    max_thresh = thresholds[max_prof]

    min_ = min(profits)
    max_ = max(profits)

    plt.figure(figsize=(15, 8))
    plt.plot(thresholds, profits, label=f'Profit Curve (with ${incentive_cost} promotion/customer)')
    plt.vlines(max_thresh, min_, max_ + 50000, linestyle='--', label='Max. Incr. Profit Thresh. : %.2f' % max_thresh)
    plt.ylabel('Profit ($)', fontsize=20)
    plt.xlabel('Probability Threshold', fontsize=20)
    plt.title('Profit Curve', fontsize=20)

    max_profit_text = "Maximum Incremental Profit: $" + "{0:,}".format(max_)
    plt.text(max_thresh + 0.02, max_ + 8000, max_profit_text, fontsize=15)

    plt.legend(loc=1)

    return None