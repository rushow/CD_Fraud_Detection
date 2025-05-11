import pandas as pd
from river import naive_bayes, tree, ensemble, linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datasets import load_dataset

from concept_drift.eddm import EDDMDriftDetector
from concept_drift.fhddm import FHDDMDriftDetector
from concept_drift.rddm import RDDMDriftDetector
from concept_drift.ftdd import FTDDDriftDetector
from concept_drift.arf import ARFDriftDetector
from concept_drift.dwm import DWMDriftDetector
from concept_drift.d3 import D3DriftDetector
from concept_drift.fpdd import FPDDDriftDetector
from concept_drift.adwin import ADWINDriftDetector
from concept_drift.kswin import KSWINDriftDetector
from concept_drift.mddm import MDDMDriftDetector
from concept_drift.wstd import WSTDDriftDetector
from concept_drift.ewma import EWMADriftDetector
from concept_drift.aue import AUEDriftDetector
from concept_drift.awe import AWEDriftDetector

from utils.load_data import (
    load_synthetic_financial_data,
    load_nooha_cc_fraud_data,
    load_european_cc_fraud_data,
    load_thomask_cc_fraud_data,
    load_bank_transaction_fraud_data
)
from utils.evaluation import evaluate_learner
from utils.visualization import plot_heatmap_auc, plot_multi_dataset_heatmaps

# Create results directory if it doesn't exist
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created directory: {results_dir}")

# Define datasets with their loading functions and names
datasets = [
    (load_synthetic_financial_data, "Synthetic Financial"),
    (load_nooha_cc_fraud_data, "Credit Card Fraud"),
    (load_european_cc_fraud_data, "European CC Fraud"),
    (load_thomask_cc_fraud_data, "Thomas K CC Fraud"),
    # (load_bank_transaction_fraud_data, "Bank Transaction Fraud")
]

# List of models and drift detectors
models = {
    'NB': naive_bayes.GaussianNB(),
    'HT': tree.HoeffdingTreeClassifier()
}

drift_detectors = {
    # Statistical-Based Drift Detectors
    'FTDD': FTDDDriftDetector(), # 2018
    'RDDM': RDDMDriftDetector(), # 2017
    'FHDDM': FHDDMDriftDetector(), # 2016 == River
    'EWMA': EWMADriftDetector(), # 2012
    'EDDM': EDDMDriftDetector(), # 2006 == River
    # Window-Based Drift Detectors
    'KSWIN': KSWINDriftDetector(), #2020 == River
    'FPDD': FPDDDriftDetector(), # 2018
    'WSTD': WSTDDriftDetector(), # 2018 
    'MDDM': MDDMDriftDetector(), #2018
    'ADWIN': ADWINDriftDetector(), # 2007 == River
    # Ensemble-Based Drift Detectors
    'ARF': ARFDriftDetector(), # 2017
    'D3': D3DriftDetector(), #2015
    'AUE': AUEDriftDetector(), # 2011
    'DWM': DWMDriftDetector(), # 2007  
    'AWE': AWEDriftDetector(), # 2003 
}

cd_detector_name = ''

# Dictionary to store results for each dataset
results = {}

# Process each dataset
for load_func, dataset_name in datasets:
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Load dataset
    try:
        df_name, X, y = load_func()

        
        # Limit to 100,000 samples for faster processing
        sample_size = min(100000, len(X))
        X = X[:sample_size]
        y = y[:sample_size]
        print(f"Using {sample_size} samples from {dataset_name}")


        # Use the entire dataset, no sampling
        print(f"Using all {len(X)} samples from {dataset_name}")
        
        # Initialize storage for this dataset
        data_auc = {}
        
        # Run models and detector
        for model_name, model in models.items():
            for detector_name, drift_detector in drift_detectors.items():
                cd_detector_name = detector_name
                print(f"Running {detector_name} with {model_name}...")
                
                # Evaluate the model
                _, _, metric, metric_f1, metric_precision, auc_value, auroc_value = evaluate_learner(
                    df_name, model, drift_detector, X, y
                )
                
                # Store results
                if model_name not in data_auc:
                    data_auc[model_name] = {}
                data_auc[model_name][detector_name] = auc_value
                
                # Print metrics
                print(f"Accuracy: {metric}")
                print(f"Precision: {metric_precision}")
                print(f"F1: {metric_f1}")
                print(f"AUC: {auc_value}")
                print(f"AUROC: {auroc_value}")
        
        # Store results for this dataset
        results[dataset_name] = pd.DataFrame.from_dict(data_auc, orient='index')
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        continue

# Create improved multi-dataset visualization
try:
    print("Creating improved multi-dataset visualization...")
    multi_fig = plot_multi_dataset_heatmaps(
        results=results,
        figsize=(18, 14),  # Larger figure size for better spacing
        cmap="RdYlBu_r",   # Color scheme that differentiates values well
        wspace=0.3,        # More horizontal space between subplots
        hspace=0.4,        # More vertical space between subplots
        save_path=os.path.join(results_dir, "improved_all_datasets_heatmaps.png")
    )
    plt.show()
except Exception as e:
    print(f"Error creating multi-dataset visualization: {str(e)}")

# Create individual heatmaps for each dataset with enhanced readability
# for dataset_name, auc_df in results.items():
#     try:
#         print(f"Creating visualization for {dataset_name}...")
        
#         # Create enhanced heatmap
#         fig, ax = plot_heatmap_auc(
#             data=auc_df,
#             figsize=(10, 6),
#             cmap="viridis",  # Use consistent colormaps or try different ones
#             title_fontsize=16,
#             annotation_fontsize=12,
#         )
        
#         # Override the default title with the dataset name
#         ax.set_title(f'{dataset_name} Dataset', 
#                     fontsize=16, 
#                     weight='bold', 
#                     pad=20)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(results_dir, f"{dataset_name.replace(' ', '_')}_heatmap.png"), dpi=300, bbox_inches='tight')
#         plt.show()
#     except Exception as e:
#         print(f"Error creating visualization for {dataset_name}: {str(e)}")

# Create summary DataFrame for the most recent detector
# summary_df = pd.DataFrame({
#     dataset_name: df[cd_detector_name].mean() 
#     for dataset_name, df in results.items()
# }, index=['Average AUC']).T

# Plot summary heatmap
# try:
#     plt.figure(figsize=(10, 6))
#     sns.heatmap(
#         summary_df,
#         annot=True,
#         fmt='.3f',
#         cmap='RdYlBu_r',
#         center=0.75,
#         vmin=0.5,
#         vmax=1.0,
#         cbar_kws={'label': 'Average AUC Score'},
#         annot_kws={'size': 12, 'weight': 'bold'},
#         linewidths=0.5
#     )
#     plt.title(f'{cd_detector_name} Performance Across Datasets', fontsize=16, weight='bold', pad=20)
#     plt.tight_layout()
#     plt.savefig(os.path.join(results_dir, f"{cd_detector_name}_summary.png"), dpi=300, bbox_inches='tight')
#     plt.show()
# except Exception as e:
#     print(f"Error creating summary heatmap: {str(e)}")

# Print summary statistics
print("\nSummary Statistics:")
print("==================")
for dataset_name, df in results.items():
    print(f"\n{dataset_name} Dataset:")
    print(f"Average AUC: {df[cd_detector_name].mean():.3f}")
    print(f"Std Dev AUC: {df[cd_detector_name].std():.3f}")