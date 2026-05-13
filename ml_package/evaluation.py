import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def willmott_index(y_obs, y_pred):
    obs_mean = np.mean(y_obs)
    num = np.sum((y_pred - y_obs) ** 2)
    den = np.sum((np.abs(y_pred - obs_mean) + np.abs(y_obs - obs_mean)) ** 2)
    if den == 0:
        return 0
    return 1 - (num / den)

def evaluate_regression(y_true, y_pred):
    return {
        'R2': max(0.0, r2_score(y_true, y_pred)),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'Willmott_d': willmott_index(y_true, y_pred)
    }

def evaluate_classification(y_true, y_pred, y_proba=None, n_classes=2, average='weighted'):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'F1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if y_proba is not None:
        try:
            if n_classes == 2:
                metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except ValueError:
            pass  # single class in split — skip ROC_AUC
    return metrics
