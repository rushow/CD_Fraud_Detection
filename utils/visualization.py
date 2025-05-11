import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
import os

def plot_heatmap_auc(
    data: pd.DataFrame,
    figsize: tuple = (12, 8),
    cmap: str = "viridis",
    title_fontsize: int = 16,
    label_fontsize: int = 14,
    annotation_fontsize: int = 12,
    rotation_xticks: int = 45,
    vmin: Optional[float] = 0.5,
    vmax: Optional[float] = 1.0,
    center: Optional[float] = 0.75,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create an enhanced heatmap visualization for AUC scores comparison.
    """
    # Set the style
    sns.set_style("white")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for missing values
    mask = np.isnan(data)
    
    # Create heatmap
    sns.heatmap(
        data=data,
        ax=ax,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        mask=mask,
        annot_kws={'size': annotation_fontsize, 'weight': 'bold'},
        linewidths=0.5,
        linecolor='white',
        cbar_kws={
            'label': 'AUC Score',
            'orientation': 'vertical',
            'shrink': 0.8,
            'pad': 0.02,
            'aspect': 30
        }
    )
    
    # Customize appearance
    ax.set_title('AUC Performance Comparison:\nModels vs Drift Detectors', 
                fontsize=title_fontsize, 
                weight='bold',
                pad=20)
    
    ax.set_xlabel('Drift Detectors', fontsize=label_fontsize, weight='bold', labelpad=10)
    ax.set_ylabel('Models', fontsize=label_fontsize, weight='bold', labelpad=10)
    
    # Rotate x-axis labels and make them bold
    plt.xticks(rotation=rotation_xticks, ha='right', fontsize=label_fontsize-2, fontweight='bold')
    plt.yticks(fontsize=label_fontsize-2, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_multi_dataset_heatmaps(
    results: Dict[str, pd.DataFrame],
    figsize: tuple = (18, 14),
    cmap: str = "viridis",
    save_path: Optional[str] = None,
    wspace: float = 0.3,
    hspace: float = 0.4
) -> plt.Figure:
    """
    Create a figure with subplots for AUC heatmaps from multiple datasets with improved spacing.
    """
    # Calculate number of rows and columns for subplots
    n_datasets = len(results)
    n_cols = min(2, n_datasets)  # Maximum 2 columns
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Create a GridSpec layout with proper spacing
    gs = fig.add_gridspec(n_rows, n_cols, wspace=wspace, hspace=hspace)
    
    # Process each dataset
    for idx, (dataset_name, auc_df) in enumerate(results.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        # Create subplot with proper position
        ax = fig.add_subplot(gs[row, col])
        
        # Create heatmap for this dataset
        sns.heatmap(
            data=auc_df,
            annot=True,
            fmt='.3f',
            cmap=cmap,
            center=0.75,
            vmin=0.5,
            vmax=1.0,
            ax=ax,
            annot_kws={'size': 11, 'weight': 'bold'},
            linewidths=0.5,
            cbar_kws={'label': 'AUC Score', 'shrink': 0.8}
        )
        
        # Set title and labels
        ax.set_title(dataset_name, fontsize=14, weight='bold', pad=10)
        ax.set_xlabel('Drift Detectors', fontsize=12, labelpad=5)
        ax.set_ylabel('Models', fontsize=12, labelpad=5)
        
        # Format tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
    
    # Add overall title
    fig.suptitle('AUC Performance Comparison Across Datasets', 
                fontsize=18, 
                weight='bold',
                y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to accommodate suptitle
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_drift_detection_timeline(
    t: List[int],
    accuracy: List[float],
    drift_points: List[int],
    dataset_name: str,
    model_name: str,
    detector_name: str,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize accuracy over time with drift detection points.
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot accuracy over time
    ax.plot(t, accuracy, color='blue', alpha=0.7, linewidth=2, label='Accuracy')
    
    # Add moving average for smoothing
    window_size = min(len(accuracy) // 10, 100)  # 10% of data or max 100 points
    window_size = max(window_size, 1)  # Ensure window size is at least 1
    if window_size > 1:
        rolling_mean = pd.Series(accuracy).rolling(window=window_size).mean()
        ax.plot(t, rolling_mean, color='darkblue', linewidth=2, label=f'Moving Avg (window={window_size})')
    
    # Mark drift detection points
    if drift_points:
        for drift_point in drift_points:
            if drift_point in t:
                idx = t.index(drift_point)
                ax.axvline(x=drift_point, color='red', linestyle='--', alpha=0.6)
                if idx < len(accuracy):
                    ax.plot(drift_point, accuracy[idx], 'ro', markersize=8)
    
    # Customize appearance
    ax.set_title(f'Accuracy Over Time with Drift Detection\n{dataset_name} | {model_name} | {detector_name}',
                fontsize=14, weight='bold', pad=20)
    
    ax.set_xlabel('Instances', fontsize=12, weight='bold', labelpad=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12, weight='bold', labelpad=10)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right', fontsize=10)
    
    # Set y-axis limits
    ax.set_ylim([0, 105])  # 0-100% with a little padding
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_performance_comparison(
    results: Dict[str, Dict[str, float]],
    metric_name: str = "AUC",
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a bar plot comparing performance metrics across datasets and detectors.
    """
    # Convert nested dictionary to DataFrame for plotting
    df = pd.DataFrame(results)
    
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    df.plot(kind='bar', ax=ax, width=0.8)
    
    # Customize appearance
    ax.set_title(f'{metric_name} Performance Comparison Across Datasets',
                fontsize=16, weight='bold', pad=20)
    
    ax.set_xlabel('Drift Detectors', fontsize=14, weight='bold', labelpad=10)
    ax.set_ylabel(metric_name, fontsize=14, weight='bold', labelpad=10)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)
    
    # Add grid and legend
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Datasets', loc='upper right', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_roc_curve(
    fpr_dict: Dict[str, List[float]],
    tpr_dict: Dict[str, List[float]],
    roc_auc_dict: Dict[str, float],
    title: str = "ROC Curve",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curves for multiple classifiers.
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve for each model
    for model_name in fpr_dict.keys():
        ax.plot(
            fpr_dict[model_name],
            tpr_dict[model_name],
            lw=2,
            label=f'{model_name} (AUC = {roc_auc_dict[model_name]:.3f})'
        )
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    # Customize appearance
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('False Positive Rate', fontsize=14, weight='bold', labelpad=10)
    ax.set_ylabel('True Positive Rate', fontsize=14, weight='bold', labelpad=10)
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    dataset_name: str,
    figsize: tuple = (15, 10),
    cmap: str = "Blues",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrices for multiple models.
    """
    # Calculate number of rows and columns for subplots
    n_models = len(confusion_matrices)
    n_cols = min(3, n_models)  # Maximum 3 columns
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Set the style
    sns.set_style("white")
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes for easier indexing
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot confusion matrix for each model
    for i, (model_name, cm) in enumerate(confusion_matrices.items()):
        if i < len(axes):
            ax = axes[i]
            
            # Create heatmap
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap=cmap,
                cbar=False,
                ax=ax,
                annot_kws={'size': 12, 'weight': 'bold'},
                linewidths=0.5,
                linecolor='white'
            )
            
            # Set title and labels
            ax.set_title(model_name, fontsize=14, weight='bold')
            ax.set_xlabel('Predicted', fontsize=12, weight='bold')
            ax.set_ylabel('Actual', fontsize=12, weight='bold')
            
            # Set tick labels
            ax.set_xticklabels(['Non-Fraud', 'Fraud'], fontsize=10)
            ax.set_yticklabels(['Non-Fraud', 'Fraud'], fontsize=10)
    
    # Hide unused subplots
    for i in range(len(confusion_matrices), len(axes)):
        axes[i].axis('off')
    
    # Add overall title
    fig.suptitle(f'Confusion Matrices - {dataset_name}', 
                fontsize=18, 
                weight='bold',
                y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to accommodate suptitle
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_drift_detector_comparison(
    drift_points_dict: Dict[str, List[int]],
    dataset_name: str,
    max_samples: int,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot drift detection points for multiple drift detectors.
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot drift points for each detector
    for i, (detector_name, drift_points) in enumerate(drift_points_dict.items()):
        # Create vertical lines at each drift point
        for drift_point in drift_points:
            ax.axvline(x=drift_point, color=f'C{i}', linestyle='--', alpha=0.5)
        
        # Create scatter points for visualization
        if drift_points:
            y_pos = np.ones(len(drift_points)) * i
            ax.scatter(drift_points, y_pos, color=f'C{i}', s=100, label=detector_name)
    
    # Customize appearance
    ax.set_title(f'Drift Detection Points Comparison - {dataset_name}',
                fontsize=16, weight='bold', pad=20)
    
    ax.set_xlabel('Instance Index', fontsize=14, weight='bold', labelpad=10)
    ax.set_ylabel('Drift Detectors', fontsize=14, weight='bold', labelpad=10)
    
    # Set axis limits
    ax.set_xlim([0, max_samples])
    ax.set_ylim([-0.5, len(drift_points_dict) - 0.5])
    
    # Set y-axis ticks
    ax.set_yticks(range(len(drift_points_dict)))
    ax.set_yticklabels(list(drift_points_dict.keys()), fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_model_metrics_radar(
    metrics_dict: Dict[str, Dict[str, float]],
    dataset_name: str,
    figsize: tuple = (10, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a radar chart comparing multiple metrics for different models.
    """
    # Get all metric names
    all_metrics = set()
    for model_metrics in metrics_dict.values():
        all_metrics.update(model_metrics.keys())
    all_metrics = sorted(list(all_metrics))
    
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(all_metrics)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], all_metrics, fontsize=12, fontweight='bold')
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each model
    for i, (model_name, metrics) in enumerate(metrics_dict.items()):
        # Complete the metrics dictionary with zeros for missing metrics
        values = [metrics.get(metric, 0) for metric in all_metrics]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    # Add title
    plt.title(f'Model Performance Metrics - {dataset_name}', 
             fontsize=16, 
             weight='bold',
             pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_class_distribution(
    class_counts: Dict[str, Dict[int, int]],
    dataset_names: List[str],
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot class distribution for multiple datasets.
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set width of bars
    bar_width = 0.35
    
    # Set position of bars on x axis
    r1 = np.arange(len(dataset_names))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    non_fraud_counts = [class_counts.get(name, {}).get(0, 0) for name in dataset_names]
    fraud_counts = [class_counts.get(name, {}).get(1, 0) for name in dataset_names]
    
    # Calculate percentages
    total_counts = [nf + f for nf, f in zip(non_fraud_counts, fraud_counts)]
    fraud_percentages = [f / t * 100 if t > 0 else 0 for f, t in zip(fraud_counts, total_counts)]
    
    # Create bars
    ax.bar(r1, non_fraud_counts, width=bar_width, label='Non-Fraud', color='skyblue')
    ax.bar(r2, fraud_counts, width=bar_width, label='Fraud', color='salmon')
    
    # Add labels and title
    ax.set_xlabel('Dataset', fontsize=14, weight='bold', labelpad=10)
    ax.set_ylabel('Count', fontsize=14, weight='bold', labelpad=10)
    ax.set_title('Class Distribution Across Datasets', fontsize=16, weight='bold', pad=20)
    
    # Set x-axis ticks
    ax.set_xticks([r + bar_width/2 for r in range(len(dataset_names))])
    ax.set_xticklabels(dataset_names, fontsize=12)
    
    # Add percentages as text on top of fraud bars
    for i, (count, percentage) in enumerate(zip(fraud_counts, fraud_percentages)):
        ax.text(r2[i], count, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Use log scale if the difference between counts is large
    max_count = max(max(non_fraud_counts), max(fraud_counts))
    min_count = min(min([c for c in non_fraud_counts if c > 0]), min([c for c in fraud_counts if c > 0]))
    if max_count / min_count > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Count (log scale)', fontsize=14, weight='bold', labelpad=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_feature_importance(
    feature_importance: Dict[str, Dict[str, float]],
    dataset_name: str,
    figsize: tuple = (12, 8),
    top_n: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance for multiple models.
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Calculate number of models
    n_models = len(feature_importance)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)
    
    # Handle single model case
    if n_models == 1:
        axes = [axes]
    
    # Plot feature importance for each model
    for i, (model_name, features) in enumerate(feature_importance.items()):
        # Sort features by importance and take top N
        sorted_features = dict(sorted(features.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        # Create horizontal bar plot
        ax = axes[i]
        sns.barplot(
            x=list(sorted_features.values()),
            y=list(sorted_features.keys()),
            ax=ax,
            palette='viridis'
        )
        
        # Set title and labels
        ax.set_title(model_name, fontsize=14, weight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        if i == 0:
            ax.set_ylabel('Features', fontsize=12)
        else:
            ax.set_ylabel('')
        
        # Add values to bars
        for j, v in enumerate(sorted_features.values()):
            ax.text(v + 0.01, j, f'{v:.3f}', va='center', fontsize=9)
    
    # Add overall title
    fig.suptitle(f'Feature Importance - {dataset_name}', 
                fontsize=16, 
                weight='bold',
                y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to accommodate suptitle
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_learning_curve(
    training_sizes: List[int],
    train_scores: List[float],
    test_scores: List[float],
    model_name: str,
    dataset_name: str,
    metric_name: str = "Accuracy",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot learning curve (performance as a function of training set size).
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot learning curve
    ax.plot(training_sizes, train_scores, 'o-', color='blue', label=f'Training {metric_name}')
    ax.plot(training_sizes, test_scores, 'o-', color='red', label=f'Validation {metric_name}')
    
    # Fill the area between train and test scores
    ax.fill_between(training_sizes, train_scores, test_scores, alpha=0.1, color='gray')
    
    # Customize appearance
    ax.set_title(f'Learning Curve: {model_name} on {dataset_name}', 
                fontsize=16, weight='bold', pad=20)
    
    ax.set_xlabel('Training Set Size', fontsize=14, weight='bold', labelpad=10)
    ax.set_ylabel(metric_name, fontsize=14, weight='bold', labelpad=10)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig