import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_fs_single(imp_df: pd.DataFrame, method_name: str):
    """Horizontal bar chart for one feature selection method (Normalized_Score)."""
    df = imp_df.sort_values("Normalized_Score", ascending=True)
    fig = px.bar(
        df, x="Normalized_Score", y="Feature", orientation="h",
        title=f"Feature Selection — {method_name}",
        color="Normalized_Score",
        color_continuous_scale=["#e0e0e0", "#1DB954"],
        text=df["Score"].round(4),
    )
    fig.update_traces(textposition="outside", textfont_size=12)
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Normalized Score (0–1)",
        yaxis_title="",
        height=max(350, len(df) * 28),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111", size=14),
        title_font=dict(size=18, color="#111111"),
    )
    fig.update_xaxes(
        gridcolor="#e0e0e0", linecolor="#999999",
        tickfont=dict(color="#111111", size=13),
        title_font=dict(color="#111111", size=14),
        range=[0, 1.15],
    )
    fig.update_yaxes(
        gridcolor="#e0e0e0", linecolor="#999999",
        tickfont=dict(color="#111111", size=13),
    )
    return fig


def plot_fs_comparison(results: dict):
    """
    Grouped bar chart comparing Normalized_Score across multiple FS methods.

    Parameters
    ----------
    results : dict mapping method_label -> imp_df (from run_feature_selection)
    """
    from plotly.subplots import make_subplots

    # Collect all features across methods (union), keep top 15 by max score
    all_features = set()
    for df in results.values():
        all_features.update(df["Feature"].tolist())

    # Score each feature by its max normalized score across methods
    feat_max = {}
    for feat in all_features:
        best = 0.0
        for df in results.values():
            row = df[df["Feature"] == feat]
            if not row.empty:
                best = max(best, row["Normalized_Score"].iloc[0])
        feat_max[feat] = best

    top_features = sorted(feat_max, key=feat_max.get, reverse=True)[:15]

    palette = ["#1DB954", "#535353", "#22C55E", "#B3B3B3", "#16A34A", "#777"]
    method_names = list(results.keys())

    fig = go.Figure()
    for i, name in enumerate(method_names):
        df = results[name]
        scores = []
        for feat in top_features:
            row = df[df["Feature"] == feat]
            scores.append(row["Normalized_Score"].iloc[0] if not row.empty else 0.0)
        fig.add_trace(go.Bar(
            name=name,
            x=top_features,
            y=scores,
            marker_color=palette[i % len(palette)],
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
            textfont=dict(size=11, color="#111111"),
        ))

    fig.update_layout(
        barmode="group",
        title="Feature Selection Comparison — Normalized Scores (Top 15)",
        title_font=dict(size=18, color="#111111"),
        height=520,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111", size=14),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=13, color="#111111"),
        ),
        margin=dict(t=100, b=80),
        xaxis=dict(
            tickangle=35,
            tickfont=dict(color="#111111", size=13),
            title_font=dict(color="#111111", size=14),
            gridcolor="#e0e0e0", linecolor="#999999",
        ),
        yaxis=dict(
            title="Normalized Score (0–1)",
            tickfont=dict(color="#111111", size=13),
            title_font=dict(color="#111111", size=14),
            gridcolor="#e0e0e0", linecolor="#999999",
            range=[0, 1.2],
        ),
    )
    return fig

def plot_correlation_matrix(df: pd.DataFrame):
    """
    Returns a Plotly figure for the correlation matrix of numerical features.
    """
    corr = df.select_dtypes(include=[np.number]).corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Matriz de Correlação"
    )
    fig.update_layout(height=600)
    return fig

def plot_interactive_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    fig = px.scatter(df, x=x_col, y=y_col, title=f"Dispersão: {x_col} vs {y_col}", trendline="ols")
    return fig

def plot_interactive_histogram(df: pd.DataFrame, col: str):
    fig = px.histogram(df, x=col, marginal="box", title=f"Distribuição: {col}")
    return fig

def plot_feature_importance(importance_df: pd.DataFrame):
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 title="Feature Importance (Feature Selection)",
                 color='Importance', color_continuous_scale='Viridis')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_regression_performance(preds_df: pd.DataFrame, model_name: str):
    """
    Returns a Plotly figure comparing observed vs predicted for regression for BOTH train and test.
    preds_df must have ['Observed', 'Predicted', 'Dataset']
    """
    fig = px.scatter(preds_df, x='Observed', y='Predicted', color='Dataset',
                     color_discrete_map={'Train': 'green', 'Test': 'red'},
                     title=f"{model_name} - Observed vs Predicted", opacity=0.7)
    
    # Linha ideal
    min_val = min(preds_df['Observed'].min(), preds_df['Predicted'].min())
    max_val = max(preds_df['Observed'].max(), preds_df['Predicted'].max())
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="black", dash="dash"))
    return fig

def plot_confusion_matrix_plotly(y_true, y_pred, classes, title="Confusion Matrix"):
    """
    Returns a Plotly figure for confusion matrix.
    """
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import numpy as np
 
    
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
 
    
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]
 
    
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)
 
    
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    str_labels = [str(l) for l in labels]
 
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
 
    
    if cm.shape[0] != len(str_labels):
        raise ValueError(
            f"Shape mismatch: cm.shape={cm.shape}, len(labels)={len(str_labels)}, "
            f"labels={labels}"
        )
 
    fig = px.imshow(
        cm, text_auto=True,
        x=str_labels, y=str_labels,
        color_continuous_scale="Blues", title=title,
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Observed")
    return fig
 


def plot_metrics_summary(res_df: pd.DataFrame, task_type: str):
    """
    2x2 subplot grouped bar chart — one panel per metric, Train vs Test side-by-side.
    Regression: R2, RMSE, MAE, Willmott_d.
    Classification: Accuracy, Precision, Recall, F1.
    """
    from plotly.subplots import make_subplots

    if task_type == "Regression":
        base_metrics = [("R2", "R²"), ("RMSE", "RMSE"),
                        ("MAE", "MAE"), ("Willmott_d", "Willmott d")]
    else:
        base_metrics = [("Accuracy", "Accuracy"), ("Precision", "Precision"),
                        ("Recall", "Recall"), ("F1", "F1")]

    # Keep only metrics present in both Train_ and Test_ columns
    metrics = [
        (base, lbl)
        for base, lbl in base_metrics
        if f"Train_{base}" in res_df.columns and f"Test_{base}" in res_df.columns
    ]
    if not metrics:
        return None

    n = len(metrics)
    rows, cols = (1, n) if n <= 2 else (2, 2)
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[lbl for _, lbl in metrics],
        vertical_spacing=0.22,
        horizontal_spacing=0.12,
    )

    TRAIN_COLOR = "#535353"   # dark grey
    TEST_COLOR  = "#1DB954"   # green

    for idx, (base, lbl) in enumerate(metrics):
        r, c = divmod(idx, cols)
        shown = idx == 0  # legend only on first subplot

        fig.add_trace(
            go.Bar(
                x=res_df["Model"],
                y=res_df[f"Train_{base}"].round(4),
                name="Train",
                marker_color=TRAIN_COLOR,
                legendgroup="Train",
                showlegend=shown,
                text=res_df[f"Train_{base}"].round(3),
                textposition="outside",
                textfont=dict(size=11, color="#111111"),
            ),
            row=r + 1, col=c + 1,
        )
        fig.add_trace(
            go.Bar(
                x=res_df["Model"],
                y=res_df[f"Test_{base}"].round(4),
                name="Test",
                marker_color=TEST_COLOR,
                legendgroup="Test",
                showlegend=shown,
                text=res_df[f"Test_{base}"].round(3),
                textposition="outside",
                textfont=dict(size=11, color="#111111"),
            ),
            row=r + 1, col=c + 1,
        )

    fig.update_layout(
        title_text="Model Performance Summary — Train vs Test",
        title_font=dict(size=20, color="#111111"),
        barmode="group",
        height=750,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111", size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.04,
            xanchor="right", x=1,
            font=dict(size=14, color="#111111"),
        ),
        margin=dict(t=100, b=60),
    )
    fig.update_xaxes(
        tickangle=30,
        tickfont=dict(color="#111111", size=13),
        title_font=dict(color="#111111", size=14),
        gridcolor="#e0e0e0",
        linecolor="#999999",
    )
    fig.update_yaxes(
        tickfont=dict(color="#111111", size=13),
        title_font=dict(color="#111111", size=14),
        gridcolor="#e0e0e0",
        linecolor="#999999",
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=16, color="#111111")

    return fig
