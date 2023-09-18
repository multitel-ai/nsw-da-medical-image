from pathlib import Path
import glob 
import json

import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
import scipy.linalg as linalg
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .dataset_util import enums
from .classifier.utils import get_test_transforms

ALLOWED_EXT = ["jpg","jpeg"]

def shorten_dataset(img_list,labels,size=1000):
    """
    Randomly selects a subset of images and their corresponding labels from a given dataset.

    Args:
    img_list (list): A list of image file paths.
    labels (list): A list of labels corresponding to the images.
    size (int, optional): The desired size of the shortened dataset. Default is 1000.

    Returns:
    tuple: A tuple containing two elements:
        - List of selected image file paths (new_img_list).
        - NumPy array of corresponding labels (new_labels).

    The function organizes the input images and labels by their associated categories and video names
    and then randomly selects images from each category and video until the desired dataset size is reached.

    Example:
    >>> img_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
    >>> labels = ["cat", "dog", ...]
    >>> new_img_list, new_labels = shorten_dataset(img_paths, labels, size=500)
    """

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
    """
    This function takes an image file path and extracts a numeric index from the
    file name. It assumes that the index is located after the string "RUN" and
    before the file extension. For example, if the image_path is "image_RUN123.jpg",
    this function will return 123 as an integer.

    Example:
    >>> img_path = "path/to/image_RUN42.jpg"
    >>> index = get_img_ind(img_path)
    >>> print(index)
    42
    """     
    img_path = Path(img_path)
    img_name = img_path.stem
    index_str = img_name.split("RUN")[1]
    index = int(index_str)
    return index

def get_dataset_name(dataset_path):
    if dataset_path.endswith("/"):
        dataset_path = dataset_path[:-1]
    dataset_name = dataset_path.split("/")[-1]
    return dataset_name

def get_imgs(root,is_orig_data,orig_annot_folder=None):
    """
    Collects image file paths, focal plane information, and labels for a dataset.

    Args:
    root (str): The root directory containing image files or metadata.
    is_orig_data (bool): Indicates whether the dataset is composed of original or synthetic data.
    orig_annot_folder (str, optional): The folder containing annotations for original data if is_orig_data is True.

    Returns:
    tuple: A tuple containing three elements:
        - List of image file paths (found_images).
        - Focal plane information (focal_plane).
        - NumPy array of labels (labels).

    Example:
    >>> root_dir = "../data/extracted/embryo_dataset/"
    >>> is_original = True
    >>> orig_annot_folder = "../data/extracted/embryo_dataset_annotation/"
    >>> images, focal_plane, labels = get_imgs(root_dir, is_original, orig_annot_folder)
    """

    found_images = []
    for ext in ALLOWED_EXT:
        found_images += glob.glob(str(root / f"*.{ext}"))

    all_labels_list = [phase.label for phase in list(enums.Phase)]

    if not is_orig_data:
   
        with open(root / "metadata.json","r") as f:
            metadata = json.load(f)

        focal_plane = metadata["focal_plane"]
        label = all_labels_list.index(metadata["phase"])
        labels = np.array([label]*len(found_images)).astype("int")
        
    else:
        
        vid_name = root.name

        annotation_path = orig_annot_folder / str(vid_name+"_phases.csv")
        if annotation_path.exists():
            phases = np.genfromtxt(annotation_path,dtype=str,delimiter=",")
            labels = np.zeros((int(phases[-1,-1])+1))-1
            for phase in phases:
                labels[int(phase[1]):int(phase[2])+1] = all_labels_list.index(phase[0])
            
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

def load_original_dataset(root,annot_folder,split_file_path,dataset_label,max_size=None,debug=False):

    #Load the json file 
    with open(split_file_path) as f:
        split_dic = json.load(f)    

    found_images = []
    labels = np.array([])
    folds = glob.glob(str(root / "*/"))
    
    for fold in folds:
        vid_name = fold.split("/")[-1] 
        if vid_name in split_dic["test"]:
            fold = Path(fold)
            found_image_fold,_,labels_fold = get_imgs(fold,True,annot_folder)
            found_images += found_image_fold
            labels = np.concatenate((labels,labels_fold),axis=0)
        
    img_list = np.array(found_images)

    label_mask = labels == dataset_label
    img_list = img_list[label_mask]
    labels = labels[label_mask]
    
    if max_size is not None and len(img_list) > max_size:
        img_list,labels = shorten_dataset(img_list,labels,max_size)

    assert len(labels) == len(img_list)
    print("Using",len(labels),"original images from label",dataset_label)

    if debug:
        with open("img_and_labels_orig.txt","w") as f:
            for path,label in zip(img_list,labels):
                print(path,label,file=f)

    return img_list,labels
           
def load_synthetic_data(root,debug=False):
    img_list,_,labels = get_imgs(root,False)

    #Labels contains the same value repeated as many times as there are images
    #In a synthetic dataset, all images have the same label
    dataset_label = labels[0]

    assert len(labels) == len(img_list)
    print("Using",len(labels),"synthetic images from label",dataset_label)

    if debug:
        with open("img_and_labels_synth.txt","w") as f:
            for path,label in zip(img_list,labels):
                print(path,label,file=f)
        
    return img_list,labels
        
class BasicImageDataset():

    def __init__(self,transform,img_list,labels):

        self.transform = transform
        self.img_list,self.labels = img_list,labels

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,idx):
        img = Image.open(self.img_list[idx])
        if self.transform is not None:
            img = self.transform(img)
        if img.shape[0] == 1:
            img = np.repeat(img,3,0)
        return img,self.labels[idx]
    
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

def compute_synth_img_metrics(model_weights_path,synth_data_path,model_architecture="resnet50",result_fold_path="../results/",orig_data_path="../data/extracted/embryo_dataset/",orig_data_annot_folder="../data/extracted/embryo_dataset_annotations/",split_file_path="split.json",debug=False,num_classes=16,max_dataset_size=5000,batch_size=50,num_workers=0):
    """
    Compute metrics for synthetic image data using a specified deep learning model.

    Args:
    model_architecture (str): The architecture of the deep learning model (can be "densenet121" or "resnet50").
    result_fold_path (str): The directory to save result files.
    model_weights_path (str): The path to the model weights file.
    debug (bool): Indicates whether to run in debug mode.
    num_classes (int): The number of classes.
    synth_data_path (str): The path to the synthetic image folder.
    orig_data_path (str): The path to the original image dataset.
    orig_data_annot_folder (str): The folder containing annotations for the original dataset.
    split_file_path (str): The path to the file specifying dataset splits ("split.json").
    max_dataset_size (int): The size of the subset that will be sampled from the original dataset.
    batch_size (int): Batch size for inference.
    num_workers (int): The number of data loader workers.

    Returns:
    None

    This function computes various metrics for synthetic image data using a specified deep learning model.
    It calculates FID (Frechet Inception Distance), entropy, and accuracy for both original and synthetic datasets
    and saves the results in a CSV file.

    Example:
    >>> model_architecture = "densenet121"
    >>> result_fold_path = "/path/to/results/"
    >>> model_weights_path = "/path/to/model/weights.pth"
    >>> debug = False
    >>> num_classes = 16
    >>> synth_data_path = "../data/synthetic/model1/run1/"
    >>> orig_data_path = "../data/extracted/embryo_dataset"
    >>> orig_data_annot_folder = "../data/extracted/embryo_dataset_annotations"
    >>> split_file_path = "split.json"
    >>> max_dataset_size = 5000
    >>> batch_size = 32
    >>> num_workers = 0
    >>> compute_synth_img_metrics(model_architecture, result_fold_path, model_weights_path, debug, num_classes,
    ... synth_data_path, orig_data_path, orig_data_annot_folder, split_file_path, max_dataset_size, batch_size,
    ... num_workers)
    """

    assert model_architecture in ["densenet121","resnet50"]

    result_fold_path = Path(result_fold_path)
    if not result_fold_path.exists():
        result_fold_path.mkdir(parents=True)

    cuda = torch.cuda.is_available()
    torch.set_grad_enabled(False)

    # Load model
    if model_weights_path is None:
        model = getattr(models,model_architecture)(weights="IMAGENET1K_V1")
        if debug:
            print("Warning: no model path provided, using imagenet weights to debug")
        else:
            raise ValueError("No model path provided")
        model_name = None
    else:
        model = getattr(models,model_architecture)(num_classes=num_classes)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        weights = torch.load(model_weights_path,map_location=device)
        model.load_state_dict(weights)
        model_name = Path(model_weights_path).stem

    model.eval()
    if cuda:
        model = model.cuda()
  
    #add a hook on the last feature layer of the model 
    #to get the feature vector
    vector_list = []
    def save_output(_,features,__):
        features = features[0]

        if debug:
            features = features[:,:10]

        vector_list.append(features[0].cpu())

    if model_architecture == "densenet121":        
        model.classifier.register_forward_hook(save_output)
    else:
        model.fc.register_forward_hook(save_output)

    stat_dic = {}

    synth_data_path = Path(synth_data_path)
    orig_data_path = Path(orig_data_path)
    orig_data_annot_folder = Path(orig_data_annot_folder)
    split_file_path = Path(split_file_path)

    img_list,labels = load_synthetic_data(synth_data_path,debug)
    synth_dataset = BasicImageDataset(get_test_transforms(),img_list,labels)
    
    dataset_label = labels[0]
    img_list,labels = load_original_dataset(orig_data_path,orig_data_annot_folder,split_file_path,dataset_label,max_dataset_size,debug)
    orig_dataset = BasicImageDataset(get_test_transforms(),img_list,labels)

    for dataset,data_dir_path in zip([orig_dataset,synth_dataset],[orig_data_path,synth_data_path]):

        dataset_name = data_dir_path.name

        mu_path = result_fold_path / f"mu_{dataset_name}_{model_name}.npy"
        sigma_path = result_fold_path / f"std_{dataset_name}_{model_name}.npy"
        entropy_path = result_fold_path / f"entropy_{dataset_name}_{model_name}.npy"
        accuracy_path = result_fold_path / f"accuracy_{dataset_name}_{model_name}.npy"

        logit_list = []
        vector_list = []
        labels_list = []

        if not mu_path.exists() or not entropy_path.exists():

            dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

            for i,(imgs,labels) in enumerate(dataloader):
                if cuda:
                    imgs = imgs.cuda()
                logit_list.append(model(imgs).cpu())
                labels_list.append(labels)

                if i > 1 and debug:
                    break
            
            labels = torch.cat(labels_list).numpy()
            vectors = torch.cat(vector_list).numpy()

            mu = np.mean(vectors, axis=0)
            std = np.cov(vectors, rowvar=False)

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

    orig_dataset = orig_data_path.name
    synth_dataset = synth_data_path.name

    fid = calculate_frechet_distance(stat_dic[orig_dataset]["mu"],stat_dic[orig_dataset]["std"],stat_dic[synth_dataset]["mu"],stat_dic[synth_dataset]["std"])
    entropy_orig_data = stat_dic[orig_dataset]["entropy"]
    entropy_synth_data = stat_dic[synth_dataset]["entropy"]

    accuracy_orig_data = stat_dic[orig_dataset]["accuracy"]
    accuracy_synth_data = stat_dic[synth_dataset]["accuracy"]

    csv_path = result_fold_path / "synth_img_metrics.csv"

    if not csv_path.exists():
        with open(csv_path,"w") as f:
            f.write("model_name,orig_data,synth_data,label,entropy_orig,entropy_synth,accuracy_orig,accuracy_synth,fid\n")

    with open(csv_path,"a") as f:
        f.write(f"{model_name},{orig_dataset},{synth_dataset},{dataset_label},{entropy_orig_data},{entropy_synth_data},{accuracy_orig_data},{accuracy_synth_data},{fid}\n")

if __name__ == "__main__":
    main()