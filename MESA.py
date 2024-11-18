"""
 # @ Author: Chaorong Chen
 # @ Create Time: 2022-06-14 17:00:56
 # @ Modified by: Chaorong Chen
 # @ Modified time: 2024-11-18 00:48:13
 # @ Description: MESA
 """

from sklearn.base import clone
from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import StratifiedKFold
from boruta import BorutaPy
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import GenericUnivariateSelect, VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


def wilcoxon(X, y):
    """
    Score function for feature selection using Wilcoxon rank-sum test.

    Args:
        X: dataframe or array of shape (n_features, n_samples)
        y: array-like of shape (n_samples,)

    Returns:
        p-values of Wilcoxon rank-sum test for each feature
    """
    return -mannwhitneyu(X[y == 0], X[y == 1])[1]


class BorutaSelector(BorutaPy):
    def __init__(self, n=10, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def fit(self, X, y):
        super().fit(X, y)
        self.indices = np.argsort(self.ranking_)[: self.n]
        return self

    def transform(self, X):
        try:
            self.ranking_
        except AttributeError:
            raise ValueError("You need to call the fit(X, y) method first.")
        try:
            return X.iloc[:, self.indices]
        except:
            return X[:, self.indices]


class MESA_modality:
    def __init__(
        self,
        random_state=0,
        boruta_estimator=RandomForestClassifier(random_state=0, n_jobs=-1),
        top_n=100,
        variance_threshold=0,
        normalization=False,
        missing=0.1,
        classifier=RandomForestClassifier(random_state=0, n_jobs=-1),
        selector=GenericUnivariateSelect(
            score_func=wilcoxon, mode="k_best", param=2000
        ),
        **kwargs
    ):
        self.random_state = random_state
        self.boruta_estimator = boruta_estimator
        self.top_n = top_n
        self.variance_threshold = variance_threshold
        self.normalization = normalization
        self.missing = missing
        self.classifier = classifier
        self.selector = selector
        for key, value in kwargs.items():
            setattr(self, key, value)
        pass

    def fit(self, X, y):
        pipeline_steps = [
            VarianceThreshold(self.variance_threshold),
            self.selector,
            BorutaSelector(
                estimator=self.boruta_estimator,
                random_state=self.random_state,
                verbose=0,
                n_estimators="auto",
                n=self.top_n,
            ),
        ]
        if self.normalization:
            pipeline_steps.insert(0, Normalizer())
        self.pipeline = make_pipeline(*pipeline_steps).fit(X, y)
        self.classifier = self.classifier.fit(self.pipeline.transform(X), y)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def transform_predict(self, X):
        return self.classifier.predict(self.pipeline.transform(X))

    def transform_predict_proba(self, X):
        return self.classifier.predict_proba(self.pipeline.transform(X))

    def get_params(self, deep=True):
        return {
            "random_state": self.random_state,
            "boruta_estimator": self.boruta_estimator,
            "top_n": self.top_n,
            "variance_threshold": self.variance_threshold,
            "normalization": self.normalization,
            "missing": self.missing,
            "classifier": self.classifier,
            "selector": self.selector,
        }


class MESA:
    def __init__(
        self,
        meta_estimator,
        random_state=0,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        **kwargs
    ):
        self.meta_estimator = meta_estimator
        self.random_state = random_state
        self.cv = cv
        self.kwargs = kwargs

    def _internal_cv(self, X, y, base_estimator, train_index, test_index):
        X_train, X_test = X[train_index, :], X[test_index, :]
        return base_estimator.fit(X_train, np.array(y)[train_index]).predict_proba(
            X_test
        )

    def _base_fit(self, X, y, base_estimator):
        def _internal_cv(train_index, test_index):
            X_train, X_test = X[train_index, :], X[test_index, :]
            return (
                clone(base_estimator)
                .fit(X_train, np.array(y)[train_index])
                .predict_proba(X_test)
            )

        base_probability = np.vstack(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(_internal_cv)(train_index, test_index)
                for train_index, test_index in self.splits
            )
        )
        return base_probability

    def fit(self, modalities, X_list, y):
        # add check parameters
        self.modalities = modalities
        self.base_estimators = [
            m.classifier.fit(X, y) for m, X in zip(modalities, X_list)
        ]
        self.splits = [
            (train_index, test_index)
            for train_index, test_index in self.cv.split(X_list[0], y)
        ]
        y_stacking = np.hstack(
            [np.array(y)[test_index] for train_index, test_index in self.splits]
        )
        base_probability = np.hstack(
            [
                self._base_fit(m.transform(X), y, clone(m.classifier))
                for m, X in zip(modalities, X_list)
            ]
        )
        self.meta_estimator.fit(base_probability, y_stacking)
        return self

    def predict(self, X_list_test):
        base_probability_test = np.hstack(
            [clf.predict_proba(X) for clf, X in zip(self.base_estimators, X_list_test)]
        )
        return self.meta_estimator.predict(base_probability_test)

    def predict_proba(self, X_list_test):
        base_probability_test = np.hstack(
            [clf.predict_proba(X) for clf, X in zip(self.base_estimators, X_list_test)]
        )
        return self.meta_estimator.predict_proba(base_probability_test)
