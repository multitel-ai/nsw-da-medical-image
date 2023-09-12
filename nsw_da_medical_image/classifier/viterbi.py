import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from nsw_da_medical_image.dataset_util import enums

def predicted_phase_plot(predicted_classes,predicted_classes_viterbi,labels,phase_list,viz_fold,video):
    all_raw_seq = [predicted_classes,predicted_classes_viterbi,labels]
    all_subplot_names = ["Raw","Viterbi","GT"]
    fig,axs = plt.subplots(len(all_raw_seq),1,figsize=(15,8))

    #Generating legend handles
    color_map = plt.get_cmap("rainbow")
    leg_handles = []
    for i in range(len(phase_list)):
        leg_handles += [axs[0].add_patch(Rectangle((0,0),0.001,0.001,label=phase_list[i],color=color_map(i/len(phase_list))))]

    for i,(raw_pred_seq,subplot_name) in enumerate(zip(all_raw_seq,all_subplot_names)):
        agr_prediction_sequence,transition_inds = make_agregated_pred_sequence(raw_pred_seq)
        axs[i] = plot_bar_plot(axs[i],agr_prediction_sequence,transition_inds,len(phase_list),color_map)
        axs[i].set_xlim(0,len(raw_pred_seq))
        axs[i].set_title(subplot_name)

    fig.legend()
    fig.savefig(os.path.join(viz_fold,f"prediction_viterbi_{video}.png"))
    plt.close()

def make_agregated_pred_sequence(predictions):
    agr_prediction_sequence = [predictions[0]]
    transition_inds = [0]
    for i in range(1,len(predictions)):
        if predictions[i] != predictions[i-1]:
            agr_prediction_sequence.append(predictions[i])
            transition_inds.append(i)

    transition_inds.append(len(predictions))

    return agr_prediction_sequence,transition_inds

def plot_bar_plot(ax,agr_prediction_sequence,start,phase_nb,cmap):
    for i in range(len(agr_prediction_sequence)):
        label_ind = agr_prediction_sequence[i]
        ax.add_patch(Rectangle((start[i],0),start[i+1]-start[i],1,color=cmap(label_ind/phase_nb)))
    return ax

#https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm
#def viterbi(y, A, B, Pi=None):
#https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_Viterbi.html
def viterbi(O,A,B,C):
    """Viterbi algorithm (log variant) for solving the uncovering problem

    Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        O (np.ndarray): Observation sequence of length N
        A (np.ndarray): State transition probability matrix of dimension I x I
        B (np.ndarray): Output probability matrix of dimension I x K
        C (np.ndarray): Initial state distribution  of dimension I

    Returns:
        S_opt (np.ndarray): Optimal state sequence of length N
        D_log (np.ndarray): Accumulated log probability matrix
        E (np.ndarray): Backtracking matrix
    """
    I = A.shape[0]    # Number of states
    N = len(O)  # Length of observation sequence
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_log = np.log(B + tiny)

    # Initialize D and E matrices
    D_log = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D_log[:, 0] = C_log + B_log[:, O[0]]

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + D_log[:, n-1]
            #print(temp_sum,B_log)
            D_log[i, n] = temp_sum.max() + B_log[i, O[n]]
            E[i, n-1] = np.argmax(temp_sum)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N-2, -1, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    return S_opt, D_log, E

def compute_emission_matrix(split_file_path,csv_prob_path,gt_dict,class_nb):

    with open(split_file_path) as json_file:
        val_videos = json.load(json_file)["val"]

    prob_pred = np.genfromtxt(csv_prob_path,dtype=str,delimiter=",")[1:]
    conf_mat = np.zeros((class_nb,class_nb))

    for row in prob_pred:
        video = row[0].split("_")[1]

        if video in val_videos:
            gt_label = gt_dict[row[0]]
            pred_label = row[1:].astype(float).argmax()
            conf_mat[gt_label,pred_label] += 1

    emission_matrix = conf_mat/conf_mat.sum(axis=1,keepdims=True)

    return emission_matrix

def compute_transition_matrix(split_json_path,annot_fold_path,class_nb):

    with open(split_json_path) as json_file:
        train_videos = json.load(json_file)["train"]

    transition_matrix = np.zeros((class_nb,class_nb))
    priors = np.zeros((class_nb))

    for video in train_videos:
        annot_file_name = video+"_phases.csv"
        annot_file_path = os.path.join(annot_fold_path,annot_file_name)

        csv = np.genfromtxt(annot_file_path,delimiter=",",dtype=str)

        for i in range(len(csv)):
            row = csv[i]
            label_ind = enums.Phase(row[0]).idx()
            transition_matrix[label_ind,label_ind] += int(row[2])-int(row[1])

            if i == 0:
                priors[label_ind] += 1
            elif i < len(csv) - 1:
                next_label_ind = enums.Phase(csv[i+1,0]).idx()
                transition_matrix[label_ind,next_label_ind] += 1
    
    transition_matrix /= transition_matrix.sum(axis=1,keepdims=True)

    priors /= priors.sum()

    return transition_matrix,priors

def get_gt_dict(ground_truth_path):
    gt_csv = np.genfromtxt(ground_truth_path,dtype=str,delimiter=",")
    keys = gt_csv[0]
    gt_csv = gt_csv[1:]
    id_ind = np.argwhere(keys=="identifier")[0][0]
    label = np.argwhere(keys=="phase-index")[0][0]
    gt_dict = {row[id_ind]:int(row[label]) for row in gt_csv}
    return gt_dict
    
def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a model using Viterbi algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--csv_prob_path", type=str, help="Path to model prediction csv (with probabilities) on validation set.")

    parser.add_argument("--split_file_path", type=str, help="Path to the split json file.")
    parser.add_argument("--annotation_fold_path", type=str, help="Path to the fold path")
    parser.add_argument("--ground_truth_path", type=str, help="Path to the ground truth csv")
    parser.add_argument("--class_nb", type=int)
    parser.add_argument("--result_fold", type=str)

    args = parser.parse_args()

    gt_dict = get_gt_dict(args.ground_truth_path)
    phase_list = list(enums.Phase)

    ############################ COMPUTES OR LOAD STATE RANSITION MATRIX AND PRIORS #############################
    trans_mat_path = os.path.join(args.result_fold,"transition_matrix.npy")
    priors_path = os.path.join(args.result_fold,"priors.npy")
    if not os.path.exists(trans_mat_path):
        print("Computing transition matrix and priors")
        transition_matrix,priors = compute_transition_matrix(args.split_file_path,args.annotation_fold_path,args.class_nb)
        np.save(trans_mat_path,transition_matrix)
        np.save(priors_path,priors)
    else:
        transition_matrix = np.load(trans_mat_path)
        priors = np.load(priors_path)

    ################################ COMPUTES OR LOAD EMISSION MATRIX ######################################
    emission_mat_path = os.path.join(args.result_fold,"emission_matrix.npy")
    if not os.path.exists(emission_mat_path):
        print("Computing confusion matrix")
        emission_matrix = compute_emission_matrix(args.split_file_path,args.csv_prob_path,gt_dict,args.class_nb)
        np.save(emission_mat_path,emission_matrix)
    else:
        emission_matrix = np.load(emission_mat_path)

    csv = np.genfromtxt(args.csv_prob_path,dtype=str,delimiter=",")[1:]
    print(len(csv[:,0]))
    _,video_names,frame_inds = zip(*list(map(lambda x:x.split("_"),csv[:,0])))
    video_set = sorted(set(video_names))
    video_names = np.array(video_names)

    frame_inds = np.array(list(map(lambda x:x.split("_")[2],csv[:,0])))
        
    acc_list = []
    acc_viterbi_list = []

    print("Starting to compute accuracy with viterbi")

    model_name = os.path.basename(args.csv_prob_path).split("_pred-prob")[0]
    viz_fold = os.path.join(args.result_fold,model_name)
    os.makedirs(viz_fold,exist_ok=True)
    
    for i,video in enumerate(video_set):

        rows = csv[video_names==video]
        inds = np.argsort(frame_inds[video_names==video].astype(int))
        sorted_rows = rows[inds]
        probs = sorted_rows[:,1:].astype(float)
        predicted_classes = probs.argmax(axis=1)
        labels = np.array([gt_dict[frame_name] for frame_name in sorted_rows[:,0]])

        accuracy = (predicted_classes==labels).mean()
        acc_list.append(accuracy)
        
        predicted_classes_viterbi,_,_ = viterbi(predicted_classes, transition_matrix, emission_matrix,priors)

        #Plot for debug
        predicted_phase_plot(predicted_classes,predicted_classes_viterbi,labels,phase_list,viz_fold,video)

        accuracy_viterbi = (predicted_classes_viterbi == labels).mean()
        acc_viterbi_list.append(accuracy_viterbi)

        print(video,i,"/",len(video_set),accuracy,accuracy_viterbi)

    accuracy = np.array(acc_list).mean()
    accuracy_viterbi = np.array(acc_viterbi_list).mean()

    csv_path = os.path.join(args.result_fold,"accuracy_viterbi.csv")
    if not os.path.exists(csv_path):
        with open(csv_path,"w") as file:
            print("model_name,accuracy,accuracy_viterbi\n",file=file) 

    with open(csv_path,"a") as file:
        print(model_name+","+str(accuracy)+","+str(accuracy_viterbi),file=file)

if __name__ == "__main__":
    main()