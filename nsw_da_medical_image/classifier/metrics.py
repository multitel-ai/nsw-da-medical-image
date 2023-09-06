import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import csv
import os

def calculate_metrics(y_true, y_pred):
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['F1-Score'] = f1_score(y_true, y_pred, average='weighted')
    
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    metrics['ROC Curve'] = (fpr, tpr)
    metrics['AUC'] = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
    metrics['Precision-Recall Curve'] = (precision, recall)
    metrics['AUC_Precision_Recall'] = auc(recall, precision)
    
    metrics['Confusion Matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics

def plot_and_save_curve(fpr, tpr, title, xlabel, ylabel, output_folder, file_name):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(output_folder, file_name))

def main(ground_truth_csv, predictions_csv, output_folder):
    # Read CSV files
    gt_df = pd.read_csv(ground_truth_csv)
    pred_df = pd.read_csv(predictions_csv)
    
    # Ensure they have the same length
    if len(gt_df) != len(pred_df):
        print("The lengths of ground truth and predictions must be the same.")
        return
    
    y_true = gt_df['class'].values
    y_pred = pred_df['class'].values
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Save metrics to CSV
    with open(os.path.join(output_folder, 'metrics.csv'), 'w') as f:
        writer = csv.writer(f)
        for key, value in metrics.items():
            if key in ['ROC Curve', 'Precision-Recall Curve']:
                continue
            writer.writerow([key, value])
    
    # Save and plot curve metrics
    plot_and_save_curve(metrics['ROC Curve'][0], metrics['ROC Curve'][1], 'ROC Curve', 'False Positive Rate', 'True Positive Rate', output_folder, 'roc_curve.png')
    np.savetxt(os.path.join(output_folder, 'roc_curve.csv'), metrics['ROC Curve'], delimiter=",")
    
    plot_and_save_curve(metrics['Precision-Recall Curve'][0], metrics['Precision-Recall Curve'][1], 'Precision-Recall Curve', 'Recall', 'Precision', output_folder, 'precision_recall_curve.png')
    np.savetxt(os.path.join(output_folder, 'precision_recall_curve.csv'), metrics['Precision-Recall Curve'], delimiter=",")
    
    # Save Confusion Matrix to CSV
    np.savetxt(os.path.join(output_folder, 'confusion_matrix.csv'), metrics['Confusion Matrix'], delimiter=",")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python <script_name>.py <ground_truth_csv_path> <predictions_csv_path> <output_folder_path>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])

