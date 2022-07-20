"""
 # @ Author: Chaorong Chen
 # @ Create Time: 2022-06-14 17:00:56
 # @ Modified by: Chaorong Chen
 # @ Modified time: 2022-07-19 21:32:26
 # @ Description: Code for MESA
 """


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    LeaveOneOut,
    train_test_split,
)
from joblib import Parallel, delayed
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score
from boruta import BorutaPy
from sklearn.linear_model import LogisticRegression
from deepforest import CascadeForestClassifier
from sklearn.svm import SVC

# Code for missing value imputation


def imputation_cv(X, train_index, test_index, ratio=0.9):
    """
    Parameters
    ----------
    X : dataframe of shape (n_features, n_samples)
        Input samples.
    train_index : list/array/tuple of
        The training set indices for the LOO split.
    test_index : list/array/tuple of
        The testing set indices for the LOO split.
    ratio : float, default=0.9
        The threshold for feature filtering. Only features have valid values for >ratio of samples are kept and then imputed.

    Returns
    ----------
    X_train_cleaned : dataframe of shape (n_train_samples, n_features)
        Cleaned, missing-value-imputed training set.
    X_test_cleaned :dataframe of shape (n_test_samples, n_features)
        Cleaned, missing-value-imputed testing datasets.
    """
    X_train_temp, X_test_temp = X.iloc[:, train_index], X.iloc[:, test_index]
    # Count valid values for each features.
    X_train_valid = X_train_temp.count(axis="columns")
    # Filter out features missing in >(1-ratio) of samples.
    X_train_seleted = np.where(X_train_valid >= X_train_temp.shape[1] * ratio)[0]
    # Imputed features left with mean values inside the training set.
    imputer = SimpleImputer(strategy="mean")
    X_train_cleaned = pd.DataFrame(
        imputer.fit_transform(X_train_temp.iloc[X_train_seleted].T.values)
    )
    # Imputed testing sets with same values as the corresponding training set.
    X_test_cleaned = pd.DataFrame(
        imputer.transform(X_test_temp.iloc[X_train_seleted].T.values)
    )
    # put Sample ID back
    X_train_cleaned.index, X_test_cleaned.index = (
        X.columns[train_index],
        X.columns[test_index],
    )
    # put feature ID back
    X_train_cleaned.columns, X_test_cleaned.columns = (
        X.index[X_train_seleted],
        X.index[X_train_seleted],
    )
    return X_train_cleaned, X_test_cleaned


# Code for sequntial backward selection


def SBS_LOO(
    X,
    y,
    estimator,
    classifiers=[],
    cv=5,
    random_state=0,
    min_feature=10,
    n_jobs=-1,
    scoring="roc_auc",
    boruta_top_n_feature=1000,
):
    """
    Parameters
    ----------
    X : dataframe of shape (n_features, n_samples)
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
        """
        Train-test spliting & Missing value imputation
        """
        X_train_temp, X_test_temp = imputation_cv(X, train_index, test_index, 0.9)
        X_train, X_test = X_train_temp.values, X_test_temp.values
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        """
        Boruta algorithm for ranking
        """
        rf_clf = RandomForestClassifier(n_jobs=-1, random_state=random_state)
        boruta_ranking = (
            BorutaPy(rf_clf, n_estimators="auto", random_state=random_state)
            .fit(X_train, y_train)
            .ranking_
        )
        rank = np.argsort(boruta_ranking)
        boruta_select = rank[:boruta_top_n_feature]
        """
        Sequential backward selection(SBS)
        """
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
                delayed(cross_val_score)(
                    estimator,
                    X_train[:, combination[i]],
                    y_train,
                    cv=cv,
                    n_jobs=-1,
                    scoring=scoring,
                )
                for i in range(len(combination))
            )
            # average peformance evaluation results for all feature subsets
            scores = np.mean(scores_, axis=1)
            # best feature subset with 'dim' feature(s)
            best = np.argmax(scores)
            # kick out the least influential feature then perform algorothm on features left
            indices = combination[best]
            all_scores.append(scores)
            all_subsets.append(combination)
            print("Dimension:", dim, " Score:", np.max(scores))
            dim -= 1
        # best feature subset at each featue size
        best_scores = [np.max(i) for i in all_scores]
        best_combination_dim = np.where(best_scores == np.max(best_scores))[0][-1]
        best_combination_loc = np.argmax(all_scores[best_combination_dim])
        feature_selected = all_subsets[best_combination_dim][best_combination_loc]
        feature_selected_all.append(feature_selected)
        print(
            "Best combination:", all_subsets[best_combination_dim][best_combination_loc]
        )
        print("Best score:", all_scores[best_combination_dim][best_combination_loc])
        print(
            "Best dimension:",
            len(all_subsets[best_combination_dim][best_combination_loc]),
        )
        """
        Summary for output
        """
        y_pred_iter = []
        for clf in classifiers:
            clf.fit(X_train[:, feature_selected], y_train)
            y_pred = clf.predict_proba(X_test[:, feature_selected])[:, 1]
            y_pred_iter.append(y_pred[0])
        y_pred_all.append(y_pred_iter)
        y_true.append(y_test[0])
    # predicted probability for each sample
    y_pred_all = np.array(y_pred_all).T
    # calculate AUC for all classifiers
    auc = [roc_auc_score(y_true, prob) for prob in y_pred_all]
    return feature_selected_all, y_true, y_pred_all, auc


# Integrate2 SBS results on different types of features


def calculate_integration(X_list, y, feature_selected, classifiers):
    """
    Parameters
    ----------
    X_list : list of dataframes of shape (n_features, n_samples)
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
    cv_index = list(cv_method.split(X_list[0].T, y))
    scaler = StandardScaler()
    for run in range(len(cv_index)):
        train_index, test_index = cv_index[run]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        X_temp = [x.iloc[:, list(fea[run])] for x, fea in zip(X_list, feature_selected)]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        X_train_temp = [
            imputation_cv(X1, train_index, test_index, 0.9)[0] for X1 in X_temp
        ]
        X_test_temp = [
            imputation_cv(X1, train_index, test_index, 0.9)[1] for X1 in X_temp
        ]
        X_train = [
            _.iloc[:, list(fea[run])].values
            for _, fea in zip(X_train_temp, feature_selected)
        ]
        X_test = [
            _.iloc[:, list(fea[run])].values
            for _, fea in zip(X_test_temp, feature_selected)
        ]
        X_train_std = []
        X_test_std = []
        for i in range(len(X_list)):
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
    # predicted probability for each sample
    y_pred_all = np.array(y_pred_all).T
    # calculate AUC for all classifiers
    auc = [roc_auc_score(y_true, prob) for prob in y_pred_all]
    return y_true, y_pred_all, auc



def SBS_LOO_3class(
    X,
    y,
    estimator,
    classifiers=[],
    cv=5,
    random_state=0,
    min_feature=10,
    n_jobs=-1,
    scoring="accuracy",
    boruta_top_n_feature=300,
):
    """
    Parameters
    ----------
    X : dataframe of shape (n_features, n_samples)
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
    acc : array-like of shape (n_classifiers,)
        Accuracy on test set, given by classifier(s) input(Same order with input).
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
        """
        Train-test spliting & Missing value imputation
        """
        X_train_temp, X_test_temp = imputation_cv(X, train_index, test_index, 0.9)
        X_train, X_test = X_train_temp.values, X_test_temp.values
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        """
        Boruta algorithm for ranking
        """
        rf_clf = RandomForestClassifier(n_jobs=-1, random_state=random_state)
        boruta_ranking = (
            BorutaPy(rf_clf, n_estimators="auto", random_state=random_state)
            .fit(X_train, y_train)
            .ranking_
        )
        rank = np.argsort(boruta_ranking)
        boruta_select = rank[:boruta_top_n_feature]
        """
        Sequential backward selection(SBS)
        """
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
                delayed(cross_val_score)(
                    estimator,
                    X_train[:, combination[i]],
                    y_train,
                    cv=cv,
                    n_jobs=-1,
                    scoring=scoring,
                )
                for i in range(len(combination))
            )
            # average peformance evaluation results for all feature subsets
            scores = np.mean(scores_, axis=1)
            # best feature subset with 'dim' feature(s)
            best = np.argmax(scores)
            # kick out the least influential feature then perform algorothm on features left
            indices = combination[best]
            all_scores.append(scores)
            all_subsets.append(combination)
            print("Dimension:", dim, " Score:", np.max(scores))
            dim -= 1
        # best feature subset at each featue size
        best_scores = [np.max(i) for i in all_scores]
        best_combination_dim = np.where(best_scores == np.max(best_scores))[0][-1]
        best_combination_loc = np.argmax(all_scores[best_combination_dim])
        feature_selected = all_subsets[best_combination_dim][best_combination_loc]
        feature_selected_all.append(feature_selected)
        print(
            "Best combination:", all_subsets[best_combination_dim][best_combination_loc]
        )
        print("Best score:", all_scores[best_combination_dim][best_combination_loc])
        print(
            "Best dimension:",
            len(all_subsets[best_combination_dim][best_combination_loc]),
        )
        """
        Summary for output
        """
        y_pred_iter = []
        for clf in classifiers:
            clf.fit(X_train[:, feature_selected], y_train)
            y_pred = clf.predict(X_test[:, feature_selected])
            y_pred_iter.append(y_pred[0])
        y_pred_all.append(y_pred_iter)
        y_true.append(y_test[0])
    # predicted probability for each sample
    y_pred_all = np.array(y_pred_all).T
    # calculate accuracy for all classifiers
    acc = [accuracy_score(y_true, prob) for prob in y_pred_all]
    return feature_selected_all, y_true, y_pred_all, acc
    
    
def calculate_integration_3class(X_list, y, feature_selected, classifiers):
    """
    Parameters
    ----------
    X_list : list of dataframes of shape (n_features, n_samples)
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
    cv_index = list(cv_method.split(X_list[0].T, y))
    scaler = StandardScaler()
    for run in range(len(cv_index)):
        train_index, test_index = cv_index[run]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        X_temp = [x.iloc[:, list(fea[run])] for x, fea in zip(X_list, feature_selected)]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        X_train_temp = [
            imputation_cv(X1, train_index, test_index, 0.9)[0] for X1 in X_temp
        ]
        X_test_temp = [
            imputation_cv(X1, train_index, test_index, 0.9)[1] for X1 in X_temp
        ]
        X_train = [
            _.iloc[:, list(fea[run])].values
            for _, fea in zip(X_train_temp, feature_selected)
        ]
        X_test = [
            _.iloc[:, list(fea[run])].values
            for _, fea in zip(X_test_temp, feature_selected)
        ]
        X_train_std = []
        X_test_std = []
        for i in range(len(X_list)):
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
            y_pred = clf.predict(X_test_combine.values)
            y_pred_iter.append(y_pred[0])
        y_pred_all.append(y_pred_iter)
        y_true.append(y_test[0])
    # predicted probability for each sample
    y_pred_all = np.array(y_pred_all).T
    # calculate AUC for all classifiers
    acc = [accuracy_score(y_true, prob) for prob in y_pred_all]
    return y_true, y_pred_all, acc