"""
  @Content: Feature selection code for MESA
  @Author: Chaorong Chen
  @Date: 2022-06-12 00:20:08
  @Last Modified by: Chaorong Chen
  @Last Modified time: 2022-06-14 16:27:03
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut, train_test_split
from joblib import Parallel, delayed
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score
from boruta import BorutaPy

# Code for sequntial backward selection 
def SBS_LOO(X,
            y,
            estimator,
            classifiers=[],
            cv=5,
            random_state=0,
            min_feature=10,
            n_jobs=-1,
            scoring='roc_auc',
            boruta_top_n_feature=1000):
    """
    Parameters
    ----------
    X : dataframe of shape (n_samples, n_features)
        Input samples.
    y : array-like of shape (n_samples,)
        Target values/labels.
    estimator : estimator object/model implementing ‘fit’
        The object to use to fit the data.
    classifiers : a list of estimator object/model implementing ‘fit’ and 'predict_proba'
        The object to use to evalutate on test set at the end.
    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            None, to use the default 5-fold cross validation,
            int, to specify the number of folds in a (Stratified)KFold,
            CV splitter,
            An iterable yielding (train, test) splits as arrays of indices.    
    random_state : int, RandomState instance or None, default=0
        Controls the pseudo random number generation for shuffling the data.
    min_feature : int, default=10
        The minimal feature size SBS should consider
    n_jobs : int, default=-1
        Number of jobs to run in parallel. When evaluating a new feature to add or remove, the cross-validation procedure is parallel over the folds. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.  
    scoring : str or callable, default='roc_auc'
        For SBS process, a str (see scikit-learn model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y) which should return only a single value.    
    boruta_top_n_feature : int, default=1000
        Features to select for SBS in the Boruta algorithm.
    
    Returns
    ----------
    feature_selected_all : a list of tuples (n_samples,)
        Features selected in each LOO iteration.
    y_true : array-like of shape (n_samples,)
        Target values/labels.
    y_pred_all : array-like of shape (n_samples, n_classifiers)
        Predicted probablity given by the classifiers(Same order with input).
    auc : array-like of shape (n_classifiers,)
        AUC on test set, given by classifier(s) input(Same order with input).
    """
    cv_method = LeaveOneOut()
    y_pred_all = []
    y_true = []
    feature_selected_all = []
    num = 1
    if boruta_top_n_feature > X.shape[1]:
        boruta_top_n_feature = X.shape[1]
    cv_index = cv_method.split(X, y)
    for train_index, test_index in cv_index:
        print("=============== No.", num, " LOO iteration ===============")
        num += 1
        '''
        Train-test spliting
        '''
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        '''
        Boruta algorithm for ranking
        '''
        rf_clf = RandomForestClassifier(n_jobs=-1, random_state=random_state)
        boruta_ranking = BorutaPy(rf_clf,
                                  n_estimators='auto',
                                  random_state=random_state).fit(
                                      X_train, y_train).ranking_
        rank = np.argsort(boruta_ranking)
        boruta_select = rank[:boruta_top_n_feature]
        '''
        Sequential backward selection(SBS)
        '''
        all_scores = []
        all_subsets = []
        dim = boruta_top_n_feature
        indices = tuple((boruta_select))
        while dim >= min_feature:
            scores = []
            subsets = []
            combination = list(combinations(indices, r=dim))
            # Parallel computation from 'joblib' package
            scores_ = Parallel(n_jobs=n_jobs)(
                delayed(cross_val_score)(estimator,
                                         X_train[:, combination[i]],
                                         y_train,
                                         cv=cv,
                                         n_jobs=-1,
                                         scoring=scoring)
                for i in range(len(combination)))
            scores = np.mean(
                scores_, axis=1
            )  # average peformance evaluation results for all feature subsets
            best = np.argmax(
                scores)  # best feature subset with 'dim' feature(s)
            indices = combination[
                best]  # kick out the least influential feature then perform algorothm on features left
            all_scores.append(scores)
            all_subsets.append(combination)
            print("Dimension:", dim, " Score:", np.max(scores))
            dim -= 1
        best_scores = [np.max(i) for i in all_scores
                       ]  # best feature subset at each featue size
        best_combination_dim = np.where(
            best_scores == np.max(best_scores))[0][-1]
        best_combination_loc = np.argmax(all_scores[best_combination_dim])
        feature_selected = all_subsets[best_combination_dim][
            best_combination_loc]
        feature_selected_all.append(feature_selected)
        print('Best combination:',
              all_subsets[best_combination_dim][best_combination_loc])
        print('Best score:',
              all_scores[best_combination_dim][best_combination_loc])
        print('Best dimension:',
              len(all_subsets[best_combination_dim][best_combination_loc]))
        '''
        Summary for output
        '''
        y_pred_iter = []
        for clf in classifiers:
            clf.fit(X_train[:, feature_selected], y_train)
            y_pred = clf.predict_proba(X_test[:, feature_selected])[:, 1]
            y_pred_iter.append(y_pred[0])
        y_pred_all.append(y_pred_iter)
        y_true.append(y_test[0])
    y_pred_all = np.array(
        y_pred_all).T  # predicted probability for each sample
    auc = [roc_auc_score(y_true, prob)
           for prob in y_pred_all]  # calculate AUC for all classifiers
    return feature_selected_all, y_true, y_pred_all, auc
    
# Combine SBS results on different types of features
def calculate_combine(X, y, feature_selected, classifiers):
    """
    Parameters
    ----------
    X : list of dataframes of shape (n_samples, n_features)
        Input samples.
    y : array-like of shape (n_samples,)
        Target values/labels.
    feature_selected :  list of tuples (n_samples) 
        Features selected for each LOO iteration (same order with X)
    classifiers : a list of estimator object/model implementing ‘fit’ and 'predict_proba'
        The object to use to evalutate on test set at the end.

    Returns
    ----------
    y_true : array-like of shape (n_samples,)
        Target values/labels.
    y_pred_all : array-like of shape (n_samples, n_classifiers)
        Predicted probablity given by the classifiers(Same order with input).
    auc : array-like of shape (n_classifiers,)
        AUC on test set, given by classifier(s) input(Same order with input).
    """
    cv_method = LeaveOneOut()
    y_pred_all = []
    y_true = []
    cv_index = list(cv_method.split(X[0], y))
    scaler = StandardScaler()
    for run in range(len(cv_index)):
        train_index, test_index = cv_index[run]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        X_temp = [
            x.iloc[:, list(fea[run])] for x, fea in zip(X, feature_selected)
        ]
        X_train = [X1.values[train_index] for X1 in X_temp]
        X_test = [X1.values[test_index] for X1 in X_temp]
        X_train_std = []
        X_test_std = []
        for i in range(len(X)):
            X_train_std.append(pd.DataFrame(scaler.fit_transform(X_train[i])))
            X_test_std.append(pd.DataFrame(scaler.transform(X_test[i])))
        print(X_test_std)
        print(X_train_std)
        X_train_combine = pd.concat(X_train_std, axis=1)
        X_train_combine.columns = np.arange(X_train_combine.shape[1])
        X_test_combine = pd.concat(X_test_std, axis=1)
        X_test_combine.columns = np.arange(X_test_combine.shape[1])
        y_pred_iter = []
        for clf in classifiers:
            clf.fit(X_train_combine.values, y_train)
            y_pred = clf.predict_proba(X_test_combine.values)[:, 1]
            y_pred_iter.append(y_pred[0])
        y_pred_all.append(y_pred_iter)
        y_true.append(y_test[0])
    y_pred_all = np.array(
        y_pred_all).T  # predicted probability for each sample
    auc = [roc_auc_score(y_true, prob)
           for prob in y_pred_all]  # calculate AUC for all classifiers
    return y_true, y_pred_all, auc
