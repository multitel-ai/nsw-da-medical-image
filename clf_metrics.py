import os
import pandas as pd
from pathlib import Path 
import argparse
from nsw_da_medical_image.classifier.metrics import calculate_metrics


def main(args):    
    assert os.path.exists(args.gt), f"Path not found: {args.data_dir}"
    assert os.path.exists(args.pred_path), f"Path not found: {args.pred_path}"
    
    for f in os.listdir(args.pred_path):
        if f.endswith('pred-prob.csv'):
            pred_prob = Path(args.pred_path) / Path(f)
        if f.endswith('pred-class.csv'):
            pred_labels = Path(args.pred_path) / Path(f)
    
    results = calculate_metrics(
      gt_csv=args.gt,
      pred_labels_csv=pred_labels,
      pred_prob_csv=pred_prob, 
      n_classes=args.n_classes
    )
    
    out_path = Path(args.pred_path)
    # Call the calculate_metrics function and capture its output
    conf_mat, norm_conf_mat, acc_balanced, prec_class, rec_class, f1_score_class, roc_auc_class, pr_auc_class = results
    
    # Save confusion matrix
    pd.DataFrame(conf_mat).to_csv(out_path / 'conf_mat.csv', index=False)
    
    # Save normalized confusion matrix
    pd.DataFrame(norm_conf_mat).to_csv(out_path / 'norm_conf_mat.csv', index=False)
    
    # Save class-wise metrics into a single CSV
    class_metrics_df = pd.DataFrame({
        'Precision': prec_class,
        'Recall': rec_class,
        'F1_Score': f1_score_class,
        'PR_AUC': pr_auc_class,
        'ROC_AUC': roc_auc_class
    })
    class_metrics_df.to_csv(out_path / 'class_metrics.csv', index=False)
    
    # Save other metrics
    other_metrics_df = pd.DataFrame({
        'Balanced_Accuracy': [acc_balanced],
    })
    other_metrics_df.to_csv(out_path / 'other_metrics.csv', index=False)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Computes metrics based on the results infered and stored in CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--gt", type=str, default='ground-truth.csv', help="Path to ground truth CSV.")
    parser.add_argument("--pred_path", type=str, 
                        help="Directory where model_pred-prob.csv and model_pred-class.csv can be found.")
    parser.add_argument("--n_classes", type=int, default=16, help="Number of classes.")

    args = parser.parse_args()
    main(args)