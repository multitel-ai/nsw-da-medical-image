from sklearn.metrics import confusion_matrix,roc_auc_score,precision_recall_curve,auc
import numpy as np
import pandas as pd


def calculate_metrics(gt_csv,pred_labels_csv,pred_prob_csv, n_classes=16):
    pred_labels = []
    true_labels= []
    prob_pred_labels = []
    prec_class = []
    rec_class = []
    f1_score_class = []
    pr_auc_class = []
    
    true_info = pd.read_csv(gt_csv)
    pred_info = pd.read_csv(pred_labels_csv)
    prob_pred_info = pd.read_csv(pred_prob_csv)
    true_dict = dict(zip(true_info["identifier"],true_info["phase-index"]))
    pred_dict = dict(zip(pred_info["id"],pred_info["class_pred"]))

    prob_pred_dict = {}
    for index, rows in prob_pred_info.iterrows():
        prob_pred_dict[rows.id] =list(rows[1:])
          
    

    for id,cl in pred_dict.items():
        true_labels.append(true_dict[id])
        pred_labels.append(cl)
        prob_pred_labels.append(prob_pred_dict[id])
    
    conf_mat = confusion_matrix(true_labels, pred_labels)
    norm_conf_mat = confusion_matrix(true_labels, pred_labels,normalize='true')
    acc_balanced = norm_conf_mat.diagonal().sum() / norm_conf_mat.sum()
    for i in range(n_classes):
        TP_c = norm_conf_mat[i, i]
        FP_c = norm_conf_mat[:, i].sum() - TP_c
        FN_c = norm_conf_mat[i, :].sum() - TP_c
        
        if TP_c + FP_c <= 0:
            prec_class.append(np.nan)
        else:
            prec_class.append(TP_c/(TP_c + FP_c))
    
        rec_class.append(TP_c / (TP_c + FN_c))
        f1_score_class.append(TP_c / (TP_c + 0.5 * FP_c + 0.5 * FN_c))
    
        true_labels_cl = [1 if (label == i) else 0 for label in true_labels]
        prob_pred_labels_cl = [prob_pred_label[i] for prob_pred_label in prob_pred_labels]
        prec_cl, rec_cl, _ = precision_recall_curve(true_labels_cl,prob_pred_labels_cl)
        pr_auc_class.append(auc(rec_cl, prec_cl))
        
    roc_auc_class = roc_auc_score(true_labels, prob_pred_labels, average = None, multi_class = 'ovr')
    
    return conf_mat, norm_conf_mat, acc_balanced, prec_class, rec_class, f1_score_class,roc_auc_class,pr_auc_class



