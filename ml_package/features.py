import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# ── Scaling ───────────────────────────────────────────────────────────────────

def scale_features(X: pd.DataFrame, method: str = "StandardScaler"):
    """
    Scale the features using the specified method.
    Options: 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'None'
    """
    if method == "StandardScaler":
        scaler = StandardScaler()
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif method == "RobustScaler":
        scaler = RobustScaler()
    elif method in ("None", None):
        return X, None
    else:
        raise ValueError(f"Scaling method {method} not supported.")

    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled, scaler


# ── Feature Selection ─────────────────────────────────────────────────────────

# Human-readable label → internal key
FEATURE_SELECTION_METHODS = {
    "ExtraTrees Importance":        "extratrees",
    "Random Forest Importance":     "random_forest",
    "Mutual Information":           "mutual_info",
    "F-test (ANOVA / F-regression)": "ftest",
    "RFE — Random Forest":          "rfe",
    "Lasso / L1 Regularization":    "lasso",
}


def _prep(X: pd.DataFrame, y: pd.Series):
    """Fill NaN and align index."""
    X_f = X.fillna(X.mean())
    idx = X_f.index.intersection(y.dropna().index)
    return X_f.loc[idx], y.loc[idx]


def _to_df(features, scores, method_name: str) -> pd.DataFrame:
    """Normalize scores → standardized output DataFrame."""
    df = pd.DataFrame({"Feature": list(features), "Score": list(scores)})
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    lo, hi = df["Score"].min(), df["Score"].max()
    df["Normalized_Score"] = (df["Score"] - lo) / (hi - lo) if hi > lo else 1.0
    df["Rank"] = range(1, len(df) + 1)
    df["Method"] = method_name
    # cumulative importance on normalized scores
    total = df["Normalized_Score"].sum()
    df["Cumulative_Importance"] = (
        df["Normalized_Score"].cumsum() / total if total > 0 else 0
    )
    return df


def _fs_extratrees(X: pd.DataFrame, y: pd.Series, task_type: str) -> pd.DataFrame:
    from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
    X_f, y_f = _prep(X, y)
    cls = ExtraTreesRegressor if task_type == "Regression" else ExtraTreesClassifier
    m = cls(n_estimators=100, random_state=42, n_jobs=-1)
    m.fit(X_f, y_f)
    return _to_df(X.columns, m.feature_importances_, "ExtraTrees")


def _fs_random_forest(X: pd.DataFrame, y: pd.Series, task_type: str) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    X_f, y_f = _prep(X, y)
    cls = RandomForestRegressor if task_type == "Regression" else RandomForestClassifier
    m = cls(n_estimators=100, random_state=42, n_jobs=-1)
    m.fit(X_f, y_f)
    return _to_df(X.columns, m.feature_importances_, "Random Forest")


def _fs_mutual_info(X: pd.DataFrame, y: pd.Series, task_type: str) -> pd.DataFrame:
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    X_f, y_f = _prep(X, y)
    fn = mutual_info_regression if task_type == "Regression" else mutual_info_classif
    scores = fn(X_f, y_f, random_state=42)
    return _to_df(X.columns, scores, "Mutual Information")


def _fs_ftest(X: pd.DataFrame, y: pd.Series, task_type: str) -> pd.DataFrame:
    from sklearn.feature_selection import f_regression, f_classif
    X_f, y_f = _prep(X, y)
    fn = f_regression if task_type == "Regression" else f_classif
    f_scores, _ = fn(X_f, y_f)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    return _to_df(X.columns, f_scores, "F-test")


def _fs_rfe(X: pd.DataFrame, y: pd.Series, task_type: str) -> pd.DataFrame:
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    X_f, y_f = _prep(X, y)
    n = X_f.shape[1]
    cls = RandomForestRegressor if task_type == "Regression" else RandomForestClassifier
    estimator = cls(n_estimators=50, random_state=42, n_jobs=-1)
    selector = RFE(estimator, n_features_to_select=max(1, n // 2), step=1)
    selector.fit(X_f, y_f)
    # Convert ranking: rank 1 = best → invert so higher = better for display
    scores = (selector.ranking_.max() + 1) - selector.ranking_
    return _to_df(X.columns, scores.astype(float), "RFE")


def _fs_lasso(X: pd.DataFrame, y: pd.Series, task_type: str) -> pd.DataFrame:
    from sklearn.linear_model import LassoCV, LogisticRegression
    from sklearn.preprocessing import StandardScaler as _SS
    X_f, y_f = _prep(X, y)
    Xs = _SS().fit_transform(X_f)
    if task_type == "Regression":
        m = LassoCV(cv=5, random_state=42, max_iter=5000)
        m.fit(Xs, y_f)
        scores = np.abs(m.coef_)
    else:
        m = LogisticRegression(penalty="l1", solver="saga", C=0.1,
                               max_iter=2000, random_state=42)
        m.fit(Xs, y_f)
        scores = np.abs(m.coef_).mean(axis=0)
    return _to_df(X.columns, scores, "Lasso / L1")


_METHOD_FNS = {
    "extratrees":   _fs_extratrees,
    "random_forest": _fs_random_forest,
    "mutual_info":  _fs_mutual_info,
    "ftest":        _fs_ftest,
    "rfe":          _fs_rfe,
    "lasso":        _fs_lasso,
}


def run_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    method_key: str,
) -> pd.DataFrame:
    """
    Run one feature selection method.

    Parameters
    ----------
    X           : Numeric feature DataFrame.
    y           : Target Series.
    task_type   : 'Regression' or 'Classification'.
    method_key  : Key from FEATURE_SELECTION_METHODS values.

    Returns
    -------
    DataFrame with columns: Feature, Score, Normalized_Score, Rank, Method,
    Cumulative_Importance.
    """
    fn = _METHOD_FNS.get(method_key)
    if fn is None:
        raise ValueError(f"Unknown method key: {method_key!r}")
    return fn(X, y, task_type)


# Legacy wrapper kept for backward compatibility
def calculate_feature_importance(
    X: pd.DataFrame, y: pd.Series, task_type: str = "Regression"
) -> pd.DataFrame:
    return _fs_extratrees(X, y, task_type).rename(
        columns={"Normalized_Score": "_norm"}
    ).assign(Importance=lambda d: d["Score"] / d["Score"].sum())[
        ["Feature", "Importance", "Cumulative_Importance"]
    ]
