from typing import Tuple, Optional

# classification metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             matthews_corrcoef, f1_score, confusion_matrix)

# regression metrics
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error)
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

def calculateClassificationMetrics(
        y_true=None, 
        y_pred=None, 
        averge="weighted",
        normalized_cm="true"):


    dict_metrics = {
        "Accuracy" : accuracy_score(y_pred=y_pred, y_true=y_true),
        "Precision" : precision_score(y_true=y_true, y_pred=y_pred, average=averge),
        "Recall" : recall_score(y_true=y_true, y_pred=y_pred, average=averge),
        "F1-score" : f1_score(y_true=y_true, y_pred=y_pred, average=averge),
        "MCC" : matthews_corrcoef(y_true=y_true, y_pred=y_pred),
        "Confusion Matrix" : confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=normalized_cm).tolist()
    }

    return dict_metrics

def calculateMetricsKFold(
        trained_metrics, 
        scoring_list,
        preffix=""):
    
    dict_metrics = {}

    for score in scoring_list:
        dict_metrics.update({
            score: np.mean(trained_metrics[f"{preffix}{score}"])
        })
    return dict_metrics

def calculateRegressionMetrics(
        y_true=None, 
        y_pred=None):
    
    df_values = pd.DataFrame()
    df_values["y_true"] = y_true
    df_values["y_pred"] = y_pred

    dict_metrics = {
        "R2" : r2_score(y_true=y_true, y_pred=y_pred),
        "MAE" : mean_absolute_error(y_true=y_true, y_pred=y_pred),
        "MSE" : mean_squared_error(y_true=y_true, y_pred=y_pred),
        "Kendall-tau" : df_values.corr(method="kendall")["y_true"]["y_pred"],
        "Pearson" : df_values.corr(method="pearson")["y_true"]["y_pred"],
        "Spearman" : df_values.corr(method="spearman")["y_true"]["y_pred"]
    }

    return dict_metrics

def plotConfusionMatrix(
        resolution: Tuple[int, int] = (1200, 800),
        dpi: int = 100,
        color_palette = 'Blues',
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        y_true: np.ndarray = None, 
        y_pred: np.ndarray = None, 
        labels: list = None):
    """
    Plot confusion matrix
    :param resolution: figure size in pixels (width, height)
    :param dpi: dots per inch (controls figure size scaling)
    :param color_palette: seaborn or matplotlib-compatible colormap
    :param title: title of the plot
    :param x_label: label for the x-axis
    :param y_label: label for the y-axis
    :param y_true: true labels
    :param y_pred: predicted labels
    :param labels: list of label names (optional)
    :return: None
    """
    if y_true is None or y_pred is None:
        raise ValueError("Both y_true and y_pred must be provided.")
    
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    # Convert resolution from pixels to inches for matplotlib
    fig_width = resolution[0] / dpi
    fig_height = resolution[1] / dpi

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    sns.heatmap(cm, annot=True, fmt='d', cmap=color_palette, 
                xticklabels=labels, yticklabels=labels)

    plt.xlabel(x_label if x_label else 'Predicted label')
    plt.ylabel(y_label if y_label else 'True label')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def plotPredictions(
        y_true, 
        y_pred,
        resolution: Tuple[int, int] = (1200, 800),
        dpi: int = 100, 
        point_size: int = 10,
        alpha: float = 1,
        color_palette = 'husl',
        title:str = None,
        xlabel: str = 'True Values',
        ylabel: str = 'Predicted Values',
        show_line: bool = True,
        line_style: str = '--',
        line_width: float = 1.5,
        line_color: str = 'gray',
        line_alpha: float = 0.7,
        line_label: Optional[str] = 'Ideal Prediction Line'
    ):

    width_pixels, height_pixels = resolution
    width_inches = width_pixels / dpi
    height_inches = height_pixels / dpi
    
    plt.figure(figsize=(width_inches, height_inches), dpi=dpi)

    # Plot scatter points
    plt.scatter(y_true, y_pred, 
                s=point_size, 
                alpha=alpha,
                color=sns.color_palette(color_palette)[0])
    
    # Plot ideal line if requested
    if show_line:
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 
                linestyle=line_style,
                linewidth=line_width,
                color=line_color,
                alpha=line_alpha,
                label=line_label)
        
        if line_label is not None:
            plt.legend()
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    return plt.gcf()