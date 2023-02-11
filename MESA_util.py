"""
 # @ Author: Chaorong Chen
 # @ Create Time: 2022-06-14 17:00:56
 # @ Modified by: Chaorong Chen
 # @ Modified time: 2023-02-11 02:24:14
 # @ Description: MESA util
 """


import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedKFold,
)
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from boruta import BorutaPy
from sklearn.linear_model import LogisticRegression

# Code for missing value imputation and dataset splitting


def MESA_preprocessing(X, train_index, test_index, ratio=1, normalization=False):
    """
    Parameters
    ----------
    X : dataframe of shape (n_features, n_samples)
        Input samples.
    train_index : list/array/tuple of
        The training set indices for the LOO split.
    test_index : list/array/tuple of
        The testing set indices for the LOO split.
    ratio : float, default = 1
        The threshold for feature filtering. Only features have valid values for > (ratio*samples) are kept and then imputed.
    normalization: boolean, default = False
        If scale dataset witt normalizer during preprocessing
    Returns
    ----------
    X_train_cleaned : dataframe of shape (n_train_samples, n_features)
        Cleaned, missing-value-imputed training set.
    X_test_cleaned :dataframe of shape (n_test_samples, n_features)
        Cleaned, missing-value-imputed testing datasets.
    """
    X_temp = X
    X_train_temp, X_test_temp = X_temp.iloc[:, train_index], X_temp.iloc[:, test_index]
    X_train_valid = X_train_temp.count(axis="columns")
    X_train_seleted = np.where(X_train_valid >= X_train_temp.shape[1] * ratio)[0]
    imputer = SimpleImputer(strategy="mean")
    if normalization:
        scaler = Normalizer()
        X_train_cleaned = pd.DataFrame(
            scaler.fit_transform(
                imputer.fit_transform(X_train_temp.iloc[X_train_seleted].T.values)
            )
        )
        X_test_cleaned = pd.DataFrame(
            scaler.transform(
                imputer.transform(X_test_temp.iloc[X_train_seleted].T.values)
            )
        )
    else:
        X_train_cleaned = pd.DataFrame(
            imputer.fit_transform(X_train_temp.iloc[X_train_seleted].T.values)
        )
        X_test_cleaned = pd.DataFrame(
            imputer.transform(X_test_temp.iloc[X_train_seleted].T.values)
        )
    X_train_cleaned.index, X_test_cleaned.index = (
        X_temp.columns[train_index],
        X_temp.columns[test_index],
    )  # put Sample ID back
    X_train_cleaned.columns, X_test_cleaned.columns = (
        X.iloc[X_train_seleted, 0],
        X.iloc[X_train_seleted, 0],
    )
    return X_train_cleaned, X_test_cleaned


# Code for MESA single modality construction
def MESA_single(
    X,
    y,
    boruta_est=RandomForestClassifier(random_state=0, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=0, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=0.9,
    normalization=False,
    multiclass=False,
):
    """
    Parameters
    ----------
    X : dataframe of shape (n_features, n_samples)
        Input samples.
    y : array-like of shape (n_samples,)
        Target values/labels.
    boruta_est : estimator object/model implementing ‘fit’ that returns the feature_importances_ attribute.
        The object to use to fit the data.
    cv : scikit-learn CV splitter, default=LeaveOneout()
        Splitter for cross-validation and performance mesrurement
    classifiers : a list of estimator object/model implementing ‘fit’ and 'predict_proba'
        The object to use to evalutate on test set at the end.
    random_state : int, RandomState instance or None, default=0
        Controls the pseudo random number generation for shuffling the data.
    boruta_top_n_feature : int, default=100
        Top-ranked feature to select after Boruta ranking
    variance_threshold: int or float
    n_jobs : int, default=-1
        Number of jobs to run in parallel. When evaluating a new feature to add or remove, the cross-validation procedure is parallel over the folds. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.


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
    cv_method = cv
    y_pred_all = []
    y_true = []
    feature_selected_all = []
    num = 1
    if boruta_top_n_feature > X.shape[0]:
        boruta_top_n_feature = X.shape[0]
    cv_index = cv_method.split(X.T, y)
    for train_index, test_index in cv_index:
        print("=============== No.", num, " LOO iteration ===============")
        num += 1
        """
        Train-test spliting & Missing value imputation
        """
        X_train, X_test = MESA_preprocessing(
            X, train_index, test_index, missing_ratio, normalization
        )
        X_train, X_test = X_train.values, X_test.values
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        variance = VarianceThreshold().fit(X_train).variances_
        variance_selected = np.where(variance > variance_threshold)[0]
        print(
            "VarianceThreshold(%s): %s/%s features filtered"
            % (
                variance_threshold,
                X_train.shape[1] - len(variance_selected),
                X_train.shape[1],
            )
        )

        """
        Boruta algorithm for ranking
        """
        boruta_ranking = (
            BorutaPy(boruta_est, n_estimators="auto", random_state=random_state)
            .fit(X_train[:, variance_selected], y_train)
            .ranking_
        )
        rank = variance_selected[np.argsort(boruta_ranking)]
        feature_selected = rank[:boruta_top_n_feature]
        feature_selected_all.append(feature_selected)

        """
        Summary for output
        """
        y_pred_iter = []
        for clf in classifiers:
            clf.fit(X_train[:, feature_selected], y_train)
            if multiclass:
                y_pred = clf.predict(X_test[:, feature_selected])
            else:
                y_pred = clf.predict_proba(X_test[:, feature_selected])
            y_pred_iter.append(y_pred[0])
        y_pred_all.append(y_pred_iter)
        y_true.append(y_test[0])
    # predicted probability/lable for each sample
    y_pred_all = np.array(y_pred_all)
    return feature_selected_all, y_true, y_pred_all


def base_prediction(estimator_list, X_list, y, train_index, test_index, random_state=0):
    num = len(X_list)
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
    X_train = [_[train_index, :] for _ in X_list]
    X_test = [_[test_index, :] for _ in X_list]
    base_probability = np.hstack(
        [
            estimator_list[_].fit(X_train[_], y_train).predict_proba(X_test[_])
            for _ in range(num)
        ]
    )
    return base_probability


def stacking_predictor(
    estimator_list,
    X_list,
    y,
    meta_estimator,
    random_state=0,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
):
    y_stacking_cv = np.hstack(
        [np.array(y)[test_index] for train_index, test_index in cv.split(X_list[0], y)]
    )
    # base_probability: sample * feature
    base_probability = np.vstack(
        Parallel(n_jobs=-1, verbose=0)(
            delayed(base_prediction)(
                estimator_list=estimator_list,
                X_list=X_list,
                y=y,
                train_index=train_index,
                test_index=test_index,
                random_state=random_state,
            )
            for train_index, test_index in cv.split(X_list[0], y)
        )
    )
    meta_estimator.fit(base_probability, y_stacking_cv)
    return meta_estimator


# Integrate2 SBS results on different types of features


def MESA_integration(
    X_list,
    y,
    feature_selected,
    cv=LeaveOneOut(),
    missing_ratio=1,
    normalization=False,
    estimator_list=[],
    random_state=0,
    meta_estimator=[],
    stacking_cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
    multiclass=False,
):
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
    cv_method = cv
    y_pred_all = []
    y_true = []
    cv_index = list(cv_method.split(X_list[0].T, y))
    # scaler = StandardScaler()
    for run in range(len(cv_index)):
        train_index, test_index = cv_index[run]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        X_train_temp = [
            MESA_preprocessing(X1, train_index, test_index, missing_ratio, normalization)[0]
            for X1 in X_list
        ]
        X_test_temp = [
            MESA_preprocessing(X1, train_index, test_index, missing_ratio, normalization)[1]
            for X1 in X_list
        ]
        X_train = [
            _.iloc[:, list(fea[run])].values
            for _, fea in zip(X_train_temp, feature_selected)
        ]
        X_test = [
            _.iloc[:, list(fea[run])].values
            for _, fea in zip(X_test_temp, feature_selected)
        ]

        meta_est = stacking_predictor(
            estimator_list=estimator_list,
            X_list=X_train,
            y=y_train,
            meta_estimator=meta_estimator,
            random_state=random_state,
            cv=stacking_cv,
        )
        base_probability_test = np.hstack(
            [
                estimator_list[_].fit(X_train[_], y_train).predict_proba(X_test[_])
                for _ in range(len(X_list))
            ]
        )
        # prob.append(meta_est.predict_proba(base_probability_test)[:, 1])
        if multiclass:
            y_pred_all.append(meta_est.predict(base_probability_test)[0])
        else:
            y_pred_all.append(meta_est.predict_proba(base_probability_test)[0])
        y_true.append(y_test[0])
    return y_true, y_pred_all


def MESA_summary(single_result, clf_num=1, multiclass=False):
    y_true = single_result[1]
    if multiclass:
        y_pred = [[_[clf] for _ in single_result[2]] for clf in range(clf_num)]
        performance = [(accuracy_score(y_true, y_pred[clf])) for clf in range(clf_num)]
    else:
        y_pred = [[_[clf][1] for _ in single_result[2]] for clf in range(clf_num)]
        performance = [(roc_auc_score(y_true, y_pred[clf])) for clf in range(clf_num)]
    return y_true, y_pred, performance


def MESA_integration_summary(integration_result, multiclass=False):
    y_ture = integration_result[0]
    if multiclass:
        y_pred = np.array(integration_result[1])
        performance = accuracy_score(y_ture, y_pred)
    else:
        y_pred = np.array(integration_result[1])[:, 1]
        performance = roc_auc_score(y_ture, y_pred)
    return y_ture, y_pred, performance
