import io
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from ml_package import models, features, evaluation

try:
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
    )
    from rich.console import Console
    _RICH = True
except ImportError:
    _RICH = False


# ── HPO helpers ───────────────────────────────────────────────────────────────

def _get_param_grid(model_name: str, task_type: str = "Regression") -> dict:
    """
    Comprehensive param distributions for RandomizedSearchCV.
    Continuous params use scipy distributions for proper log/uniform sampling.
    """
    from scipy.stats import loguniform, uniform, randint as ri

    # ── Tree ensembles ────────────────────────────────────────────────────────
    if model_name == "Random Forest":
        return {
            "n_estimators": ri(100, 1000),
            "max_depth": [None, 5, 10, 15, 20, 30],
            "min_samples_split": ri(2, 20),
            "min_samples_leaf": ri(1, 10),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
            "bootstrap": [True, False],
        }
    if model_name == "Extra Trees":
        return {
            "n_estimators": ri(100, 1000),
            "max_depth": [None, 5, 10, 15, 20, 30],
            "min_samples_split": ri(2, 20),
            "min_samples_leaf": ri(1, 10),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        }
    if model_name == "Gradient Boosting":
        return {
            "learning_rate": loguniform(0.005, 0.3),
            "n_estimators": ri(100, 600),
            "max_depth": ri(2, 9),
            "subsample": uniform(0.6, 0.4),
            "min_samples_split": ri(2, 20),
            "min_samples_leaf": ri(1, 10),
            "max_features": ["sqrt", "log2", None],
        }
    if model_name == "HistGradientBoosting":
        return {
            "learning_rate": loguniform(0.005, 0.3),
            "max_iter": ri(100, 1000),
            "max_depth": [None, 3, 5, 7, 10, 15],
            "min_samples_leaf": ri(10, 100),
            "l2_regularization": loguniform(0.0001, 10.0),
            "max_bins": [63, 127, 255],
        }
    if model_name == "AdaBoost":
        return {
            "n_estimators": ri(50, 500),
            "learning_rate": loguniform(0.01, 2.0),
        }
    if model_name == "Decision Tree":
        return {
            "max_depth": [None, 3, 5, 7, 10, 15, 20],
            "min_samples_split": ri(2, 30),
            "min_samples_leaf": ri(1, 15),
            "max_features": ["sqrt", "log2", None],
        }

    # ── Boosting libraries ────────────────────────────────────────────────────
    if model_name == "XGBoost":
        return {
            "learning_rate": loguniform(0.005, 0.3),
            "n_estimators": ri(100, 1000),
            "max_depth": ri(3, 11),
            "subsample": uniform(0.5, 0.5),
            "colsample_bytree": uniform(0.5, 0.5),
            "reg_alpha": loguniform(0.001, 10.0),
            "reg_lambda": loguniform(0.1, 10.0),
            "min_child_weight": ri(1, 10),
            "gamma": uniform(0.0, 5.0),
        }
    if model_name == "LightGBM":
        return {
            "learning_rate": loguniform(0.005, 0.3),
            "n_estimators": ri(100, 1000),
            "num_leaves": ri(20, 200),
            "min_child_samples": ri(5, 100),
            "feature_fraction": uniform(0.5, 0.5),
            "bagging_fraction": uniform(0.5, 0.5),
            "bagging_freq": ri(1, 8),
            "reg_alpha": loguniform(0.001, 10.0),
            "reg_lambda": loguniform(0.001, 10.0),
        }
    if model_name == "CatBoost":
        return {
            "learning_rate": loguniform(0.005, 0.3),
            "iterations": ri(100, 1000),
            "depth": ri(3, 11),
            "l2_leaf_reg": ri(1, 10),
            "bagging_temperature": uniform(0.0, 1.0),
            "random_strength": uniform(0.0, 10.0),
        }

    # ── SVM ───────────────────────────────────────────────────────────────────
    if model_name == "SVM (RBF)":
        grid = {"C": loguniform(0.01, 1000.0), "gamma": ["scale", "auto"]}
        if task_type == "Regression":
            grid["epsilon"] = loguniform(0.001, 1.0)
        return grid
    if model_name in ("LinearSVR", "LinearSVC"):
        return {"C": loguniform(0.01, 100.0)}

    # ── KNN ───────────────────────────────────────────────────────────────────
    if model_name == "KNN":
        return {
            "n_neighbors": ri(3, 30),
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
        }

    # ── MLP ───────────────────────────────────────────────────────────────────
    if model_name == "MLP":
        return {
            "hidden_layer_sizes": [
                (64,), (128,), (256,),
                (64, 64), (128, 64), (128, 128), (256, 128), (256, 128, 64),
            ],
            "activation": ["relu", "tanh"],
            "alpha": loguniform(1e-5, 0.1),
            "learning_rate_init": loguniform(1e-4, 0.01),
        }

    # ── Regularised regression ────────────────────────────────────────────────
    if model_name in ("Ridge", "Bayesian Ridge"):
        return {"alpha": loguniform(0.001, 1000.0)}
    if model_name == "Lasso":
        return {"alpha": loguniform(1e-5, 10.0)}
    if model_name == "ElasticNet":
        return {"alpha": loguniform(1e-5, 10.0), "l1_ratio": uniform(0.05, 0.90)}
    if model_name == "Huber Regressor":
        return {"epsilon": uniform(1.05, 3.0), "alpha": loguniform(1e-5, 10.0)}

    # ── Linear classification ─────────────────────────────────────────────────
    if model_name == "Logistic Regression":
        return {"C": loguniform(0.001, 100.0)}
    if model_name == "Ridge Classifier":
        return {"alpha": loguniform(0.001, 1000.0)}

    return {}  # LinearRegression, GaussianNB, LDA, QDA — no meaningful HPO


def _run_random_search(model, param_grid: dict, X_train, y_train):
    search = RandomizedSearchCV(
        model, param_grid, n_iter=25, cv=5, random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def _run_optuna(model, model_name: str, X_train, y_train, task_type: str, n_trials: int):
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna not installed. Run: pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        # ── Tree ensembles ────────────────────────────────────────────────────
        if model_name == "Random Forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20, 30]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            }
        elif model_name == "Extra Trees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20, 30]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
            }
        elif model_name == "Gradient Boosting":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
        elif model_name == "HistGradientBoosting":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 1000),
                "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 7, 10, 15]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
                "max_bins": trial.suggest_categorical("max_bins", [63, 127, 255]),
            }
        elif model_name == "AdaBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
            }
        elif model_name == "Decision Tree":
            params = {
                "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 7, 10, 15, 20]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }

        # ── Boosting libraries ────────────────────────────────────────────────
        elif model_name == "XGBoost":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            }
        elif model_name == "LightGBM":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
        elif model_name == "CatBoost":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "depth": trial.suggest_int("depth", 3, 10),
                "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            }

        # ── SVM ───────────────────────────────────────────────────────────────
        elif model_name == "SVM (RBF)":
            params = {
                "C": trial.suggest_float("C", 0.01, 1000.0, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            }
            if task_type == "Regression":
                params["epsilon"] = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
        elif model_name in ("LinearSVR", "LinearSVC"):
            params = {"C": trial.suggest_float("C", 0.01, 100.0, log=True)}

        # ── KNN ───────────────────────────────────────────────────────────────
        elif model_name == "KNN":
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
            }

        # ── MLP ───────────────────────────────────────────────────────────────
        elif model_name == "MLP":
            params = {
                "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [
                    (64,), (128,), (256,),
                    (64, 64), (128, 64), (128, 128), (256, 128), (256, 128, 64),
                ]),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 0.1, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 0.01, log=True),
            }

        # ── Regularised regression ────────────────────────────────────────────
        elif model_name in ("Ridge", "Bayesian Ridge"):
            params = {"alpha": trial.suggest_float("alpha", 1e-3, 1000.0, log=True)}
        elif model_name == "Lasso":
            params = {"alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True)}
        elif model_name == "ElasticNet":
            params = {
                "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
            }
        elif model_name == "Huber Regressor":
            params = {
                "epsilon": trial.suggest_float("epsilon", 1.05, 4.0),
                "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
            }

        # ── Linear classification ─────────────────────────────────────────────
        elif model_name == "Logistic Regression":
            params = {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
        elif model_name == "Ridge Classifier":
            params = {"alpha": trial.suggest_float("alpha", 1e-3, 1000.0, log=True)}

        else:
            # LinearRegression, GaussianNB, LDA, QDA — no meaningful HPO
            return 0.0

        m = clone(model)
        m.set_params(**params)
        scoring = "r2" if task_type == "Regression" else "accuracy"
        cv_folds = min(3, len(X_train))
        scores = cross_val_score(
            m, X_train, y_train, cv=cv_folds, scoring=scoring,
            n_jobs=-1, error_score=0.0,
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_model = clone(model)
    try:
        if study.best_params:
            best_model.set_params(**study.best_params)
    except ValueError:
        pass  # all trials failed — fit with default params
    best_model.fit(X_train, y_train)
    return best_model, study


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_training_pipeline(
    df: pd.DataFrame,
    features_cols: list,
    target_col: str,
    task_type: str = "Regression",
    scaler_option: str = "StandardScaler",
    use_gridsearch: bool = False,
    test_size: float = 0.3,
    selected_models: list = None,
    hpo_method: str = "none",
    n_trials: int = 30,
):
    """
    End-to-end training pipeline.

    Parameters
    ----------
    hpo_method  : "none" | "random_search" | "optuna"
    n_trials    : Optuna trials per model (ignored for random_search).

    Returns
    -------
    results_df, predictions_df, trained_models, optuna_studies
    optuna_studies is None unless hpo_method="optuna".
    """
    # backward compat: use_gridsearch=True → random_search
    if use_gridsearch and hpo_method == "none":
        hpo_method = "random_search"

    # ── 1. Prepare data ──────────────────────────────────────────────────────
    X = df[features_cols]
    y = df[target_col]

    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # ── 2. Scaling ───────────────────────────────────────────────────────────
    X_scaled, _ = features.scale_features(X, method=scaler_option)

    # ── 3. Split ─────────────────────────────────────────────────────────────
    if task_type == "Classification":
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            # Fallback: class too rare to stratify (< 2 samples)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

    # ── 4. Model dictionary ──────────────────────────────────────────────────
    if task_type == "Regression":
        model_dict = models.get_regression_models()
    else:
        model_dict = models.get_classification_models()
        n_classes = len(np.unique(y))
        avg_method = "binary" if n_classes == 2 else "weighted"

    if selected_models:
        model_dict = {k: v for k, v in model_dict.items() if k in selected_models}

    results_list = []
    predictions_list = []
    trained_models = {}
    optuna_studies = {} if hpo_method == "optuna" else None

    hpo_label = {
        "none": "no HPO",
        "random_search": "RandomizedSearchCV",
        "optuna": f"Optuna TPE · {n_trials} trials",
    }.get(hpo_method, hpo_method)

    # ── 5. Training loop ─────────────────────────────────────────────────────
    def _train_all(progress=None, rich_task=None):
        for name, model in model_dict.items():
            if progress:
                progress.update(rich_task, description=f"[bold cyan]{name:<28}[/]")
            try:
                if hpo_method in ("random_search", "optuna") and hasattr(model, "get_params"):
                    if hpo_method == "random_search":
                        param_grid = _get_param_grid(name, task_type)
                        if param_grid:
                            model = _run_random_search(model, param_grid, X_train, y_train)
                        else:
                            model.fit(X_train, y_train)
                    else:
                        model, study = _run_optuna(
                            model, name, X_train, y_train, task_type, n_trials
                        )
                        try:
                            if study is not None and study.best_params:
                                optuna_studies[name] = study
                        except ValueError:
                            pass  # no completed trials — skip study storage
                else:
                    model.fit(X_train, y_train)

                trained_models[name] = model

                y_tr_pred = model.predict(X_train)
                y_te_pred = model.predict(X_test)

                y_tr_proba = y_te_proba = None
                if task_type == "Classification" and hasattr(model, "predict_proba"):
                    y_tr_proba = model.predict_proba(X_train)
                    y_te_proba = model.predict_proba(X_test)

                if task_type == "Regression":
                    tr_metrics = evaluation.evaluate_regression(y_train, y_tr_pred)
                    te_metrics = evaluation.evaluate_regression(y_test, y_te_pred)
                else:
                    tr_metrics = evaluation.evaluate_classification(
                        y_train, y_tr_pred, y_tr_proba, n_classes, avg_method
                    )
                    te_metrics = evaluation.evaluate_classification(
                        y_test, y_te_pred, y_te_proba, n_classes, avg_method
                    )

                row = {"Model": name}
                for k, v in tr_metrics.items():
                    row[f"Train_{k}"] = v
                for k, v in te_metrics.items():
                    row[f"Test_{k}"] = v
                results_list.append(row)

                predictions_list.append(pd.concat([
                    pd.DataFrame({
                        "Model": name, "Dataset": "Train",
                        "Observed": np.array(y_train), "Predicted": np.array(y_tr_pred),
                    }),
                    pd.DataFrame({
                        "Model": name, "Dataset": "Test",
                        "Observed": np.array(y_test), "Predicted": np.array(y_te_pred),
                    }),
                ]))

                if progress:
                    progress.console.print(
                        f"  [green]✓[/green] [white]{name}[/white]"
                    )

            except Exception as exc:
                if progress:
                    progress.console.print(f"  [red]✗[/red] [white]{name}[/white] — {exc}")
                else:
                    print(f"[WARNING] Could not train model '{name}': {exc}")

            if progress:
                progress.advance(rich_task)

    if _RICH:
        _console = Console(stderr=True)
        _console.rule(
            f"[bold green]ML Package Platform[/] · {task_type} · {hpo_label}"
        )
        with Progress(
            SpinnerColumn(style="green"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=36, style="green", complete_style="bright_green"),
            MofNCompleteColumn(),
            TextColumn("·"),
            TimeElapsedColumn(),
            TextColumn("·"),
            TimeRemainingColumn(),
            console=_console,
            transient=False,
        ) as prog:
            rich_task = prog.add_task("Starting…", total=len(model_dict))
            _train_all(progress=prog, rich_task=rich_task)
        _console.rule("[bold green]Training complete[/]")
    else:
        _train_all()

    results_df = pd.DataFrame(results_list)
    sort_col = "Test_R2" if task_type == "Regression" else "Test_Accuracy"
    if sort_col in results_df.columns:
        results_df = results_df.sort_values(by=sort_col, ascending=False)

    if not predictions_list:
        raise ValueError(
            "All models failed to train. Possible causes: target column has only one "
            "class represented in the test set, or too few samples for cross-validation. "
            "Check that your dataset has at least ~30 rows and the target has multiple classes."
        )
    final_preds_df = pd.concat(predictions_list, ignore_index=True)
    return results_df, final_preds_df, trained_models, optuna_studies


def model_to_bytes(model) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()
