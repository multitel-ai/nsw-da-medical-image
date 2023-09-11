
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import scipy.linalg as linalg
from PIL import Image
import glob 
import json
import sys 

from nsw_da_medical_image.classifier.utils import get_test_transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

ALLOWED_EXT = ["jpg","jpeg"]

LABELS_LIST = ["tPB2","tPNa","tPNf","t2","t3","t4","t5","t6","t7","t8","t9+","tM","tSB","tB","tEB","tHB"]

def shorten_dataset(img_list,labels,size=1000):

    img_dic = {}
    for label,img_path in zip(labels,img_list):

        video_name = img_path.split("/")[-2]

        if label not in img_dic:
            img_dic[label] = {}

        if video_name not in img_dic[label]:
            img_dic[label][video_name] = []

        img_dic[label][video_name].append(img_path)
    
    new_labels = []
    new_img_list = []

    while len(new_img_list) < size:

        for label in img_dic:
            
            for video_name in img_dic[label]:

                if len(new_img_list) < size:

                    if len(img_dic[label][video_name]) == 0:
                        continue

                    ind = np.random.randint(0,len(img_dic[label][video_name]),size=1)[0]

                    new_img_list.append(img_dic[label][video_name][ind])
                    new_labels.append(label)

                    img_dic[label][video_name].pop(ind)

    return new_img_list,np.array(new_labels)

def get_img_ind(img_path):
    return int(os.path.splitext(os.path.basename(img_path))[0].split("RUN")[1])

def get_dataset_name(dataset_path):
    if dataset_path.endswith("/"):
        dataset_path = dataset_path[:-1]
    dataset_name = dataset_path.split("/")[-1]
    return dataset_name

def get_imgs(root,is_orig_data,orig_annot_folder=None):
    found_images = []
    for ext in ALLOWED_EXT:
        found_images += glob.glob(os.path.join(root,"*."+ext))

    if not is_orig_data:
   
        with open(os.path.join(root,"metadata.json"),"r") as f:
            metadata = json.load(f)

        focal_plane = metadata["focal_plane"]
        label = LABELS_LIST.index(metadata["phase"])
        labels = np.array([label]*len(found_images)).astype("int")
        
    else:
        
        if root.endswith("/"):
            root = root[:-1]
        vid_name = root.split("/")[-1]

        annotation_path = os.path.join(orig_annot_folder,vid_name+"_phases.csv")
        if os.path.exists(annotation_path):
            phases = np.genfromtxt(annotation_path,dtype=str,delimiter=",")
            labels = np.zeros((int(phases[-1,-1])+1))-1
            for phase in phases:
                labels[int(phase[1]):int(phase[2])+1] = LABELS_LIST.index(phase[0])
            
            img_and_labels = []
            for img_path in found_images:
                img_ind = get_img_ind(img_path)
                if img_ind < len(labels) and labels[img_ind] != -1:
                    img_and_labels.append((img_path,labels[img_ind]))
            
            #Sort is just for easier debugging
            img_and_labels = sorted(img_and_labels,key=lambda x: get_img_ind(x[0]))
            found_images,labels = zip(*img_and_labels)

        else:
            raise ValueError("No metadata.json or corresponding phases.csv found. Root="+root)

        focal_plane = None

    return found_images,focal_plane,labels

def getitem(idx,img_list,transform,labels):
    img = Image.open(img_list[idx])
    if transform is not None:
        img = transform(img)
    if img.shape[0] == 1:
        img = np.repeat(img,3,0)
    return img,labels[idx]

class OrigImageFolder():
    
    def __init__(self,root,transform,orig_annot_folder,split_file_path,dataset_label,max_size=None,debug=False):

        self.root = root
        self.transform = transform
        self.dataset_label = dataset_label

        #Load the json file 
        with open(split_file_path) as f:
            split_dic = json.load(f)    
    
        found_images = []
        labels = np.array([])
        folds = glob.glob(os.path.join(root,"*/"))
        for fold in folds:
            vid_name = fold.split("/")[-2]
            if vid_name in split_dic["test"]:
                found_image_fold,_,labels_fold = get_imgs(fold,True,orig_annot_folder)
                found_images += found_image_fold
                labels = np.concatenate((labels,labels_fold),axis=0)
            
        self.labels = labels
        self.img_list = np.array(found_images)

        label_mask = self.labels == dataset_label
        self.img_list = self.img_list[label_mask]
        self.labels = self.labels[label_mask]
        
        if max_size is not None and len(self.img_list) > max_size:
            self.img_list,self.labels = shorten_dataset(self.img_list,self.labels,max_size)

        assert len(self.labels) == len(self.img_list)
        print("Using",len(self.labels),"original images from label",dataset_label)

        if debug:
            with open("img_and_labels_orig.txt","w") as f:
                for path,label in zip(self.img_list,labels):
                    print(path,label,file=f)
                    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,idx):
        return getitem(idx,self.img_list,self.transform,self.labels)

class SynthImageFolder():

    def __init__(self,root,transform,debug=False):

        self.root = root
        self.transform = transform
        
        found_images = []
        labels = np.array([])
        folds = glob.glob(os.path.join(root,"*/"))
        #for fold in folds:
        found_images,_,labels = get_imgs(root,False)

        #Labels contains the same value repeated as many times as there are images
        #In a synthetic dataset, all images have the same label
        self.dataset_label = labels[0]

        self.labels = labels
        self.img_list = found_images

        assert len(self.labels) == len(self.img_list)
        print("Using",len(self.labels),"synthetic images from label",self.dataset_label)

        if debug:
            with open("img_and_labels_synth.txt","w") as f:
                for path,label in zip(self.img_list,labels):
                    print(path,label,file=f)

    def __len__(self):
        return len(self.img_list)

    def get_label(self):
        return self.dataset_label

    def __getitem__(self,idx):
        return getitem(idx,self.img_list,self.transform,self.labels)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    #https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py#L152
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_data_path", type=str,help="Path to the original data. Mandatory")
    parser.add_argument("--synth_data_path", type=str,help="Path to the synthetic data. Mandatory.")    
    parser.add_argument("--orig_data_annot_folder",type=str,help="Path to the folder containing the 'XXX_phases.csv' files.")
    parser.add_argument("--result_fold_path",type=str)
    parser.add_argument("--split_file_path",type=str,help="Path to the split.json file to only run inference on the test data.")

    parser.add_argument("--model_weights_path", type=str,help="Path to the model. Mandatory except in debug mode, in which case imagenet weights are used.")    
    parser.add_argument("--model_architecture",type=str)


    parser.add_argument("--debug",action="store_true",help="Debug mode. Only uses the first dimensions of the features and only runs a few batches.")
    parser.add_argument("--val_batch_size",type=int,default=50)
    parser.add_argument("--num_workers",type=int,default=0)
    parser.add_argument("--num_classes",type=int,default=16)
    parser.add_argument("--max_dataset_size",type=int,default=5000)

    args = parser.parse_args()

    assert args.model_architecture in ["densenet121","resnet50"]

    if not os.path.exists(args.result_fold_path):
        os.makedirs(args.result_fold_path)

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

    if args.model_architecture == "densenet121":        
        model.classifier.register_forward_hook(save_output)
    else:
        model.fc.register_forward_hook(save_output)

    stat_dic = {}

    synth_dataset = SynthImageFolder(args.synth_data_path,get_test_transforms(),debug=args.debug)
    dataset_label = synth_dataset.get_label()
    orig_dataset = OrigImageFolder(args.orig_data_path,get_test_transforms(),args.orig_data_annot_folder,args.split_file_path,dataset_label=dataset_label,max_size=args.max_dataset_size,debug=args.debug)

    for dataset,data_dir_path in zip([orig_dataset,synth_dataset],[args.orig_data_path,args.synth_data_path]):

        dataset_name = get_dataset_name(data_dir_path)
        mu_path = args.result_fold_path+f"/mu_{dataset_name}_{model_name}.npy"
        sigma_path = args.result_fold_path+f"/std_{dataset_name}_{model_name}.npy"
        entropy_path = args.result_fold_path+f"/entropy_{dataset_name}_{model_name}.npy"
        accuracy_path = args.result_fold_path+f"/accuracy_{dataset_name}_{model_name}.npy"

        logit_list = []
        vector_list = []
        labels_list = []

        if not os.path.exists(mu_path) or not os.path.exists(entropy_path):

            dataloader = DataLoader(dataset,batch_size=args.val_batch_size,shuffle=False,num_workers=args.num_workers)

            for i,(imgs,labels) in enumerate(dataloader):
                if cuda:
                    imgs = imgs.cuda()
                logit_list.append(model(imgs).cpu())
                labels_list.append(labels)

                if i > 1 and args.debug:
                    break
            
            labels = torch.cat(labels_list).numpy()
            vectors = torch.cat(vector_list).numpy()

            mu = np.mean(vectors, axis=0)
            std = np.cov(vectors, rowvar=False)

            #Saves mu and std in npy files 
            np.save(mu_path,mu)
            np.save(sigma_path,std)

            logits = torch.cat(logit_list,dim=0)
            scores = torch.softmax(logits,dim=-1)
            scores = torch.clamp(scores,np.finfo(float).eps,1)
            entropy = (-scores*torch.log(scores)).sum(dim=1).mean(dim=0).numpy()
            np.save(entropy_path,entropy)

            predictions = torch.argmax(logits,dim=1).numpy()
            accuracy = (predictions == labels).astype(float).mean()
            np.save(accuracy_path,accuracy)

        else:
            mu = np.load(mu_path)
            std = np.load(sigma_path)
            entropy = np.load(entropy_path)
            accuracy = np.load(accuracy_path)

        stat_dic[dataset_name] = {"mu":mu,"std":std,"entropy":entropy,"accuracy":accuracy}

    orig_dataset = get_dataset_name(args.orig_data_path)
    synth_dataset = get_dataset_name(args.synth_data_path)

    fid = calculate_frechet_distance(stat_dic[orig_dataset]["mu"],stat_dic[orig_dataset]["std"],stat_dic[synth_dataset]["mu"],stat_dic[synth_dataset]["std"])
    entropy_orig_data = stat_dic[orig_dataset]["entropy"]
    entropy_synth_data = stat_dic[synth_dataset]["entropy"]

    accuracy_orig_data = stat_dic[orig_dataset]["accuracy"]
    accuracy_synth_data = stat_dic[synth_dataset]["accuracy"]

    csv_path = args.result_fold_path+"/img_generator_metrics.csv"

    #if csv does not exists, create it with header 
    if not os.path.exists(csv_path):
        with open(csv_path,"w") as f:
            f.write("model_name,orig_data,synth_data,label,entropy_orig,entropy_synth,accuracy_orig,accuracy_synth,fid\n")

    with open(csv_path,"a") as f:
        f.write(f"{model_name},{orig_dataset},{synth_dataset},{dataset_label},{entropy_orig_data},{entropy_synth_data},{accuracy_orig_data},{accuracy_synth_data},{fid}\n")

if __name__ == "__main__":
    main()