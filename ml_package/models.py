"""
This module contains the definitions and dictionaries of all supported Machine Learning models
for both Regression and Classification tasks.
"""

from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, HistGradientBoostingRegressor,
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, HistGradientBoostingClassifier
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, LinearSVR, SVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor,
    LogisticRegression, RidgeClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Opcionais
_HAS_XGB = _HAS_LGBM = _HAS_CAT = False
try:
    from xgboost import XGBRegressor, XGBClassifier
    _HAS_XGB = True
except ImportError:
    pass

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    pass

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    _HAS_CAT = True
except ImportError:
    pass


def get_regression_models():
    """
    Returns a dictionary of un-fitted regression models.
    """
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=300, max_features="sqrt", random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, max_features="sqrt", bootstrap=False, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(learning_rate=0.05, max_iter=500, random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
        "SVM (RBF)": SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale", max_iter=-1),
        "LinearSVR": LinearSVR(C=1.0, epsilon=0.1, max_iter=5000, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=42),
        "Multlinear regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1e-3, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=1e-3, max_iter=5000),
        "Bayesian Ridge": BayesianRidge(max_iter=300),
        "Huber Regressor": HuberRegressor(max_iter=5000)
    }

    if _HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    if _HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    if _HAS_CAT:
        models["CatBoost"] = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_state=42, verbose=0)
        
    return models


def get_classification_models():
    """
    Returns a dictionary of un-fitted classification models.
    """
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.05, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "LinearSVC": LinearSVC(C=1.0, max_iter=5000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Ridge Classifier": RidgeClassifier(),
        "Gaussian NB": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis()
    }

    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric='logloss')
    if _HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
    if _HAS_CAT:
        models["CatBoost"] = CatBoostClassifier(iterations=500, learning_rate=0.05, random_state=42, verbose=0)

    return models
