import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

def compare_metrics(directories, output_dir):
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Stores all metrics data for all models
    all_metrics = {}
    directories_names = []
    
    for dir_name in directories:
        model_metrics = {}
        
        # Read and store each metric for each model
        model_metrics['conf_mat'] = pd.read_csv(os.path.join(dir_name, 'conf_mat.csv'))
        model_metrics['norm_conf_mat'] = pd.read_csv(os.path.join(dir_name, 'norm_conf_mat.csv'))
        model_metrics['class_metrics'] = pd.read_csv(os.path.join(dir_name, 'class_metrics.csv'))
        model_metrics['other_metrics'] = pd.read_csv(os.path.join(dir_name, 'other_metrics.csv'))
        
        all_metrics[Path(dir_name).name] = model_metrics
        directories_names.append(Path(dir_name).name)
    
    directories = directories_names
    
    # Example plots to compare models
    
    # 1. Bar plot for balanced accuracy
    plt.figure(figsize=(10, 6))
    bar_data = [all_metrics[model]['other_metrics']['Balanced_Accuracy'][0] for model in directories]
    plt.bar(directories, bar_data)
    plt.xlabel('Model')
    plt.ylabel('Balanced Accuracy')
    plt.title('Comparing Balanced Accuracy Across Models')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'balanced_accuracy_comparison.png'))
    plt.close()
    
    # 2. Line plot for class-wise F1 Score
    plt.figure(figsize=(10, 6))
    for model in directories:
        plt.plot(all_metrics[model]['class_metrics']['F1_Score'], label=f"{model} F1 Score")
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('Comparing F1 Score Across Classes and Models')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_score_comparison.png'))
    plt.close()
    
    # 3. Bar plot for class-wise Precision
    fig, axs = plt.subplots(1, len(directories), figsize=(15, 6), sharey=True)
    if len(directories) == 1:
        axs = [axs]

    for idx, model in enumerate(directories):
        axs[idx].bar(range(len(all_metrics[model]['class_metrics']['Precision'])), 
                     all_metrics[model]['class_metrics']['Precision'], 
                     alpha=0.5, label=f"{model} Precision")
        axs[idx].set_xlabel('Class')
        axs[idx].set_title(f"{model} Precision")
    axs[0].set_ylabel('Precision')
    plt.suptitle('Comparing Precision Across Classes and Models')
    plt.subplots_adjust(wspace=0.4, top=0.85)  # Adjust the spacing to prevent clipping of titles
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(output_dir, 'precision_comparison.png'))
    plt.close()
    
    # 4. Bar plot for class-wise Recall
    fig, axs = plt.subplots(1, len(directories), figsize=(15, 6), sharey=True)
    if len(directories) == 1:
        axs = [axs]
        
    for idx, model in enumerate(directories):
        axs[idx].bar(range(len(all_metrics[model]['class_metrics']['Recall'])), 
                     all_metrics[model]['class_metrics']['Recall'], 
                     alpha=0.5, label=f"{model} Recall")
        axs[idx].set_xlabel('Class')
        axs[idx].set_title(f"{model} Recall")
    axs[0].set_ylabel('Recall')
    plt.suptitle('Comparing Recall Across Classes and Models')
    plt.subplots_adjust(wspace=0.4, top=0.85)  # Adjust the spacing to prevent clipping of titles
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(output_dir, 'recall_comparison.png'))
    plt.close()
    
    # 5. Bar plot for mean ROC_AUC across classes
    plt.figure(figsize=(10, 6))
    mean_roc_auc_data = [all_metrics[model]['class_metrics']['ROC_AUC'].mean() for model in directories]
    plt.bar(directories, mean_roc_auc_data)
    plt.xlabel('Model')
    plt.ylabel('Mean ROC_AUC')
    plt.title('Comparing Mean ROC_AUC Across Classes and Models')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_roc_auc_comparison.png'))
    plt.close()

    # 6. Bar plot for mean PR_AUC across classes
    plt.figure(figsize=(10, 6))
    mean_pr_auc_data = [all_metrics[model]['class_metrics']['PR_AUC'].mean() for model in directories]
    plt.bar(directories, mean_pr_auc_data)
    plt.xlabel('Model')
    plt.ylabel('Mean PR_AUC')
    plt.title('Comparing Mean PR_AUC Across Classes and Models')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_pr_auc_comparison.png'))
    plt.close()
    
