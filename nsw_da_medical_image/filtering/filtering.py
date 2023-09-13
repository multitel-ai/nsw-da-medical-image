import torch
import argparse
import torchvision.models as models
import os
from filtering_utils import OrigImageFolder,SynthImageFolder
import numpy as np
from nsw_da_medical_image.classifier.utils import get_test_transforms
from torch.utils.data import DataLoader
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_data_path", type=str,help="Path to the original data. Mandatory")
    parser.add_argument("--synth_data_path", type=str,help="Path to the synthetic data. Mandatory.")    
    parser.add_argument("--orig_data_annot_folder",type=str,help="Path to the folder containing the 'XXX_phases.csv' files.")
    parser.add_argument("--result_fold_path",type=str)
    parser.add_argument("--split_file_path",type=str,help="Path to the split.json file to only run inference on the test data.")

    parser.add_argument("--model_weights_path", type=str,help="Path to the model. Mandatory except in debug mode, in which case imagenet weights are used.")    
    parser.add_argument("--model_architecture",type=str, default="densenet121")


    parser.add_argument("--debug",action="store_true",help="Debug mode. Only uses the first dimensions of the features and only runs a few batches.")
    parser.add_argument("--val_batch_size",type=int,default=50)
    parser.add_argument("--num_workers",type=int,default=0)
    parser.add_argument("--num_classes",type=int,default=16)
    parser.add_argument("--max_dataset_size",type=int,default=5000)

    args = parser.parse_args()

    assert args.model_architecture in ["densenet121","resnet50"]

    cuda = torch.cuda.is_available()
    torch.set_grad_enabled(False)

    # Load model
    if args.model_weights_path is None:
        model = getattr(models,args.model_architecture)(weights="IMAGENET1K_V1")
        if args.debug:
            print("Warning: no model path provided, using imagenet weights to debug")
        else:
            raise ValueError("No model path provided")
        model_name = None
    else:
        model = getattr(models,args.model_architecture)(num_classes=args.num_classes)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        weights = torch.load(args.model_weights_path,map_location=device)
        model.load_state_dict(weights)
        model_name = os.path.splitext(os.path.basename(args.model_weights_path))[0]
    model.eval()
    if cuda:
        model = model.cuda()


    #add a hook on the last feature layer of the densenet121 model 
    #to get the feature vector
    vector_list = []

    def save_output(_,features,__):
        features = features[0]

        if args.debug:
            features = features[:,:10]
        
        vector_list.append(features[0].cpu())

    synth_dataset = SynthImageFolder(args.synth_data_path,get_test_transforms(),debug=args.debug)
    dataset_label = synth_dataset.get_label()
    orig_dataset = OrigImageFolder(args.orig_data_path,get_test_transforms(),args.orig_data_annot_folder,args.split_file_path,dataset_label=dataset_label,max_size=args.max_dataset_size,debug=args.debug)
    orig_data_path = args.orig_data_path
    synth_data_path = args.synth_data_path
    

    def get_vectors(dataset, data_dir_path): 

        vector_list = []

        if args.model_architecture == "densenet121":        
            last_layer_hook = model.classifier.register_forward_hook(save_output)
        else:
            last_layer_hook = model.fc.register_forward_hook(save_output)

        

        dataloader = DataLoader(dataset,batch_size=args.val_batch_size,shuffle=False,num_workers=args.num_workers)

        for i,(imgs,labels) in enumerate(dataloader):
            if cuda:
                imgs = imgs.cuda()
                
                if i > 1 and args.debug:
                    break
        vectors = torch.cat(vector_list).numpy()

        return vectors

    
    #Commpute mean and covariance matrix for the original dataset

    original_vectors = get_vectors(orig_dataset, orig_data_path)

    mu = np.mean(original_vectors, axis=0)
    cov_matrix = np.cov(original_vectors, rowvar=False)
    
    


    #TODO : iterate over synthetic dataset and ca
    synthetic_vectors = get_vectors(synth_dataset, synth_data_path)
    
    print(f"Final embedding length : {len(synthetic_vectors[0])}")

    likely_list = []

    for vector in synthetic_vectors:
        m_dist_x = np.dot((vector-mu).transpose(),np.linalg.inv(cov_matrix))
        m_dist_x = np.dot(m_dist_x, (vector-mu))
        likely_list.append(1-stats.chi2.cdf(m_dist_x, len(vector)))