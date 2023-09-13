import os
import glob
import json
import numpy as np
from PIL import Image
import scipy.linalg as linalg

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