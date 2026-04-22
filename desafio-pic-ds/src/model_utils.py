import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, classification_report
import shap
from typing import Any
from IPython.display import display

def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, title_prefix: str = "Model"):
    """
    Plota as curvas ROC e Precision-Recall lado a lado.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'{title_prefix} - Receiver Operating Characteristic')
    ax1.legend(loc="lower right")

    # PR Curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    # Encontrar threshold que maximiza F1
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    ax2.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    ax2.scatter(recall[best_idx], precision[best_idx], marker='o', color='red', label=f'Max F1 (Thresh={best_thresh:.2f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'{title_prefix} - Precision-Recall Curve')
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

    return best_thresh

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    """
    Imprime métricas de avaliação focadas no recall.
    """
    print("=== Relatório de Classificação ===")
    print(classification_report(y_true, y_pred))
    print(f"F1 Score (Macro): {f1_score(y_true, y_pred, average='macro'):.4f}")

def plot_shap_summary(model: Any, X_transformed: pd.DataFrame, feature_names: list) -> None:
    """
    Gera o SHAP summary plot.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # Adapt to SHAP version (LightGBM returns list for classification or single array for regression/binary depending on objective)
    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1] # For binary class 1
    else:
        shap_vals_to_plot = shap_values

    shap.summary_plot(shap_vals_to_plot, X_transformed, feature_names=feature_names)

def plot_shap_force(model: Any, X_transformed: pd.DataFrame, feature_names: list, instance_idx: int = 0) -> None:
    """
    Gera um force plot local do SHAP.
    """
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed.iloc[[instance_idx]])
    expected_value = explainer.expected_value

    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1][0]
        exp_val = expected_value[1] if isinstance(expected_value, list) else expected_value
    else:
        shap_vals_to_plot = shap_values[0]
        exp_val = expected_value

    display(shap.force_plot(exp_val, shap_vals_to_plot, X_transformed.iloc[instance_idx], feature_names=feature_names))

def plot_lift_curve(y_true: pd.Series, y_prob: pd.Series) -> None:
    """
    Plota a curva de Lift manual.
    """
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values(by='y_prob', ascending=False).reset_index(drop=True)

    df['cumulative_data'] = (df.index + 1) / len(df)
    df['cumulative_positives'] = df['y_true'].cumsum()
    df['cumulative_positives_pct'] = df['cumulative_positives'] / df['y_true'].sum()

    # Baseline (Random)
    baseline = df['cumulative_data']

    # Lift
    df['lift'] = df['cumulative_positives_pct'] / baseline

    plt.figure(figsize=(10, 6))
    plt.plot(df['cumulative_data'] * 100, df['lift'], label='Model Lift', lw=2)
    plt.plot([0, 100], [1, 1], 'k--', label='Baseline (Lift=1)', lw=2)

    plt.xlabel('% of Instances Sorted by Probability')
    plt.ylabel('Lift')
    plt.title('Lift Curve (Gain Chart Representation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Print gain at top 20%
    idx_20 = int(len(df) * 0.2)
    gain_20 = df.loc[idx_20, 'lift']
    print(f"Lift at Top 20% of cases: {gain_20:.2f}x better than random selection.")
