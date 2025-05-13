import time
from functools import partial
from math import log2
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn import clone
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sktime.classification.dummy import DummyClassifier
from sktime.regression.dummy import DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

from inspector import call_with_required
from metrics import adjusted_r2

CLASSIFIERS = {
    'LogisticRegression': OneVsRestClassifier(LogisticRegression()),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'XGBRegressor': XGBClassifier(),
    'LGBMRegressor': LGBMClassifier(verbose=-1),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'GaussianNB': GaussianNB(),
    'DummyClassifier': DummyClassifier(),
    # 'MLPClassifier': MLPClassifier(),
}

REGRESSORS = {
    'LinearRegression': LinearRegression(),
    'RidgeRegression': Ridge(),
    'LassoRegression': Lasso(),
    'ElasticNetRegression': ElasticNet(),

    'KNeighborsRegressor': KNeighborsRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'XGBRegressor': XGBRegressor(),
    'LGBMRegressor': LGBMRegressor(verbose=-1),
    'DummyRegressor': DummyRegressor(),
    # 'MLPRegressor': MLPRegressor(),

}

CLASSIFIER_METRICS = {
    'Accuracy': accuracy_score,
    'Balanced Accuracy': balanced_accuracy_score,
    'F1': partial(f1_score, average='weighted'),
    'ROC AUC': partial(roc_auc_score, multi_class='ovo'),
}

REGRESSOR_METRICS = {
    'R2': r2_score,
    'Adjusted R2': adjusted_r2,
    'RMSE': root_mean_squared_error,
}

classifier_remove_svc = ['SVC', 'LinearSVC', 'NuSVC']
classifier_remove_all = ['LinearSVC', 'SGDClassifier', 'MLPClassifier', 'LogisticRegression', 'LogisticRegressionCV',
                         'SVC', 'CalibratedClassifierCV', 'PassiveAggressiveClassifier', 'LabelPropagation',
                         'LabelSpreading',
                         'RandomForestClassifier', 'GradientBoostingClassifier', 'QuadraticDiscriminantAnalysis',
                         'HistGradientBoostingClassifier', 'RidgeClassifierCV', 'RidgeClassifier',
                         'AdaBoostClassifier',
                         'ExtraTreesClassifier', 'KNeighborsClassifier', 'BaggingClassifier', 'BernoulliNB',
                         'LinearDiscriminantAnalysis', 'GaussianNB', 'NuSVC', 'DecisionTreeClassifier',
                         'NearestCentroid', 'ExtraTreeClassifier', 'CheckingClassifier', 'DummyClassifier',
                         'CategoricalNB', 'StackingClassifier', 'XGBClassifier', 'LGBMClassifier', 'Perceptron']
classifier_keep_base = ['LogisticRegression', 'DecisionTreeClassifier', 'XGBClassifier', 'LGBMClassifier',
                        'KNeighborsClassifier', 'GaussianNB', 'DummyClassifier']
classifier_keep_all = ['LogisticRegression', 'DecisionTreeClassifier', 'XGBClassifier', 'LGBMClassifier',
                       'KNeighborsClassifier', 'GaussianNB', 'DummyClassifier']
classifier_keep_two = ['LogisticRegression', 'XGBClassifier']


def is_probability_metric(name):
    return name in ['ROC AUC', 'roc_auc_score', 'log_loss', 'brier_score_loss', 'average_precision_score']


class ModelEvaluator:
    def __init__(self, models_dict: dict, metrics: dict):
        """
        参数:
            models_dict: dict[str, sklearn.base.BaseEstimator]
                模型名称到模型对象的映射（必须是已初始化好的模型）
            metrics: dict[str, callable]
                指标名称到 sklearn.metrics 函数的映射
        """
        self.models_dict = models_dict
        self.metrics = metrics

    def run(self, X_train, X_test, y_train, y_test, fold_index=None):
        """
        运行所有模型并返回指标 DataFrame

        参数:
            X_train, X_test, y_train, y_test: 训练与测试数据
            fold_index: int | None，表示当前是第几折（可选）

        返回:
            pd.DataFrame，行是模型名或模型名_k，列是指标名
        """
        results = []

        for name, model in self.models_dict.items():
            # Clone models to prevent state contamination
            cloned_model = clone(model)

            start_time = time.time()
            cloned_model.fit(X_train, y_train)
            y_pred = cloned_model.predict(X_test)
            y_score = cloned_model.predict_proba(X_test) if hasattr(cloned_model, 'predict_proba') else None
            end_time = time.time()

            y_true = y_test
            local_params = {k: v for k, v in locals().items() if k != 'self'}

            row = {}
            model_id = f"{name}_{fold_index}" if fold_index is not None else name
            row['model'] = model_id
            for metric_name, metric_func in self.metrics.items():
                try:
                    score = call_with_required(metric_func, local_params)
                except Exception as e:
                    score = None
                row[metric_name] = score
            row['Time_Taken'] = end_time - start_time
            results.append(row)

        df = pd.DataFrame(results)
        df.set_index('model', inplace=True)
        return df


class LazyPredictor:
    def __init__(self,
                 is_regression: bool = True,
                 lazy_params: dict = None,

                 models_to_add: dict[str, callable] = None,
                 models_to_delete: list[str] = None,
                 models_to_keep: dict[str, callable] = None,

                 metrics_to_add: dict[str, callable] = None,
                 metrics_to_delete: list[str] = None,
                 metrics_to_keep: dict[str, callable] = None,

                 max_k_folds: int = 100,
                 random_state: int = 42,
                 ):
        self.is_regression = is_regression
        self.lazy_params = lazy_params

        self.models_to_add = models_to_add
        self.models_to_delete = models_to_delete
        self.models_to_keep = models_to_keep

        self.metrics_to_add = metrics_to_add
        self.metrics_to_delete = metrics_to_delete
        self.metrics_to_keep = metrics_to_keep

        self.max_k_folds = max_k_folds
        self.random_state = random_state

        self._init_model()

    def _init_model(self):
        selected_models = self._select_models()
        selected_metrics = self._select_metrics()

        self.model_evaluator = ModelEvaluator(selected_models, selected_metrics)

    def _select_models(self):
        if self.is_regression:
            final_models = REGRESSORS
        else:
            final_models = CLASSIFIERS

        if self.models_to_add is not None:
            final_models.update(self.models_to_add)

        if self.models_to_delete is not None:
            final_models = {k: v for k, v in final_models.items() if k not in self.models_to_delete}

        if self.models_to_keep is not None:
            final_models = self.models_to_keep

        return final_models

    def _select_metrics(self):
        if self.is_regression:
            final_metrics = REGRESSOR_METRICS
        else:
            final_metrics = CLASSIFIER_METRICS

        if self.metrics_to_add is not None:
            final_metrics.update(self.metrics_to_add)

        if self.metrics_to_delete is not None:
            final_metrics = {k: v for k, v in final_metrics.items() if k not in self.metrics_to_delete}

        if self.metrics_to_keep is not None:
            final_metrics = self.metrics_to_keep

        return final_metrics

    def _get_cv(self, n_samples):
        num_folds = min(self.max_k_folds, max(2, round(1 + log2(n_samples))))
        return StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=self.random_state) \
            if not self.is_regression else KFold(n_splits=num_folds, shuffle=True, random_state=self.random_state)

    def run(self, x, y, path=None):
        """
        执行 LazyPredict 分析并将结果保存为 Excel。
        """
        self._init_model()

        cv = self._get_cv(x.shape[0])
        result = pd.DataFrame()

        for i, (train_idx, test_idx) in enumerate(cv.split(x, y if not self.is_regression else None)):
            print(f"lazypredict: Running Fold {i}/{cv.n_splits}")

            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            df = self.model_evaluator.run(x_train, x_test, y_train, y_test, fold_index=i)
            df.reset_index(inplace=True)
            result = pd.concat([result, df], ignore_index=True)

        result.sort_values(by='model', inplace=True)

        summary_df = result.copy()
        summary_df['base_model'] = summary_df['model'].str.replace(r'_\d+$', '', regex=True)
        numeric_cols = summary_df.select_dtypes(include='number').columns
        summary_df = summary_df.groupby('base_model')[numeric_cols].mean()

        if path is None:
            path = Path.cwd()
        if not isinstance(path, Path):
            path = Path(path)
        with pd.ExcelWriter(path / 'lazypredict_results.xlsx') as writer:
            result.to_excel(writer, sheet_name='All Folds', index=False)
            summary_df.to_excel(writer, sheet_name='Summary')
        print(f"✅ Results saved to {path}")


if __name__ == '__main__':
    from sklearn.datasets import load_diabetes, load_iris
    from sklearn.model_selection import train_test_split

    # 分类
    iris = load_iris()
    X_clf, y_clf, = iris.data, iris.target
    predictor_clf = LazyPredictor(is_regression=False,
                                  # models_to_delete=classifier_remove_all,
                                  # models_to_keep=classifier_keep_two,
                                  )
    predictor_clf.run(X_clf, y_clf)

    # 回归
    diabetes = load_diabetes()

    X_reg, y_reg = diabetes.data, diabetes.target
    predictor_reg = LazyPredictor()
    predictor_reg.run(X_reg, y_reg)
