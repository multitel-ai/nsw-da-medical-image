import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import csv
import os
import sys

def calculate_metrics(y_true, y_prob, n_classes):
    metrics = {}
    metrics['Confusion Matrix'] = confusion_matrix(y_true, np.argmax(y_prob, axis=1))

    # Binarize the true labels
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    metrics['ROC Curve'] = (fpr, tpr)
    metrics['AUC'] = roc_auc

    return metrics

def save_metrics_to_csv(metrics, output_folder):
    with open(os.path.join(output_folder, 'auc.csv'), 'w') as f:
        writer = csv.writer(f)
        for key, value in metrics['AUC'].items():
            writer.writerow([key, value])

    np.savetxt(os.path.join(output_folder, 'confusion_matrix.csv'), metrics['Confusion Matrix'], delimiter=",")

def plot_curves(fpr, tpr, output_folder):
    for i in range(len(fpr)):
        plt.figure()
        plt.plot(fpr[i], tpr[i], color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve of class {i}')
        plt.savefig(os.path.join(output_folder, f'roc_curve_class_{i}.png'))

def main(ground_truth_csv, predictions_csv, softmax_csv, output_folder):
    gt_df = pd.read_csv(ground_truth_csv)
    pred_df = pd.read_csv(predictions_csv)
    softmax_df = pd.read_csv(softmax_csv)
    val_gt_df = pd.read_csv('/App/code/eval-gt.csv')
    
    # Align dataframes based on common IDs
    common_ids = set(gt_df['identifier']).intersection(set(pred_df['id']))
    gt_df = gt_df[gt_df['identifier'].isin(common_ids)]
    pred_df = pred_df[pred_df['id'].isin(common_ids)]
    softmax_df = softmax_df[softmax_df['id'].isin(common_ids)]

    # Sort and realign based on IDs
    gt_df.sort_values('identifier', inplace=True)
    pred_df.sort_values('id', inplace=True)
    softmax_df.sort_values('id', inplace=True)

    y_true = gt_df['phase-index'].values
    y_prob = softmax_df.loc[:, softmax_df.columns.str.startswith('class_prob_')].values

    n_classes = y_prob.shape[1]

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_prob, n_classes)

    # Save metrics to CSV
    save_metrics_to_csv(metrics, output_folder)

    # Plot and save ROC curves
    plot_curves(metrics['ROC Curve'][0], metrics['ROC Curve'][1], output_folder)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python <script_name>.py <ground_truth_csv_path> <predictions_csv_path> <softmax_csv_path> <output_folder_path>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
