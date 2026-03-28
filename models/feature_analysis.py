"""
SHAP-based feature importance analysis for the football match prediction models.
Provides global and per-class explanations using TreeExplainer.
"""

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def analyze_feature_importance_shap(model, X_test, feature_names, max_display=20):
    """
    Generate SHAP values and summary figures for a tree-based model.

    Args:
        model: Trained tree-based classifier (XGBoost, LightGBM, etc.)
        X_test: Test feature array (numpy or DataFrame), shape (n_samples, n_features)
        feature_names: List of feature name strings
        max_display: Max number of features shown in summary plots

    Returns:
        tuple: (bar_fig, class_figs, shap_values, importance_df)
            bar_fig      – matplotlib figure: overall feature importance bar chart
            class_figs   – list of matplotlib figures, one per outcome class
            shap_values  – raw SHAP values (list of arrays or 3-D array)
            importance_df – DataFrame ranked by mean absolute SHAP value
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is not installed. Run: pip install shap")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Convert to numpy if needed
    if hasattr(X_test, 'values'):
        X_arr = X_test.values
    else:
        X_arr = np.array(X_test)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_arr)

    # shap_values may be a list (one array per class) or a 3-D ndarray
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    elif isinstance(shap_values, list):
        shap_list = shap_values
    else:
        shap_list = [shap_values]

    # --- Overall importance bar chart ---
    fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
    if len(shap_list) > 1:
        # Mean absolute SHAP across classes
        mean_abs = np.mean([np.abs(sv) for sv in shap_list], axis=0)
    else:
        mean_abs = np.abs(shap_list[0])

    feature_importance = mean_abs.mean(axis=0)
    top_idx = np.argsort(feature_importance)[::-1][:max_display]
    ax_bar.barh(
        [feature_names[i] for i in reversed(top_idx)],
        feature_importance[list(reversed(top_idx))],
        color='steelblue',
    )
    ax_bar.set_xlabel('Mean |SHAP value|')
    ax_bar.set_title(f'Top {max_display} Feature Importances (SHAP)')
    plt.tight_layout()

    # --- Per-class figures ---
    class_names = ['Home Win', 'Draw', 'Away Win']
    class_figs = []
    for i, sv in enumerate(shap_list):
        fig, ax = plt.subplots(figsize=(10, 8))
        class_importance = np.abs(sv).mean(axis=0)
        top_c = np.argsort(class_importance)[::-1][:max_display]
        ax.barh(
            [feature_names[j] for j in reversed(top_c)],
            class_importance[list(reversed(top_c))],
            color=['#2ecc71', '#f39c12', '#e74c3c'][i % 3],
        )
        ax.set_xlabel('Mean |SHAP value|')
        label = class_names[i] if i < len(class_names) else f'Class {i}'
        ax.set_title(f'Feature Impact — {label}')
        plt.tight_layout()
        class_figs.append(fig)

    # --- Importance DataFrame ---
    overall_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_list], axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_SHAP': overall_importance,
    }).sort_values('Mean_SHAP', ascending=False).reset_index(drop=True)

    return fig_bar, class_figs, shap_values, importance_df
