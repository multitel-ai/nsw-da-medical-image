
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

ALLOWED_EXT = ["jpg","jpeg"]

def get_dataset_name(dataset_path):
    if dataset_path.endswith("/"):
        dataset_path = dataset_path[:-1]
    dataset_name = dataset_path.split("/")[-1]
    return dataset_name

def get_imgs(root):
    found_images = []
    for ext in ALLOWED_EXT:
        found_images += glob.glob(os.path.join(root,"*."+ext))
    return found_images

class SynthImageFolder():

    def __init__(self,root,transform=None):

        self.root = root
        self.transform = transform
        
        found_images = get_imgs(root)

        if len(found_images) == 0:
            print("Multiple folders mode")
            folds = glob.glob(os.path.join(root,"*/"))
            for fold in folds:
                found_images += get_imgs(fold)
        else:
            print("Single folder mode")

        self.img_list = found_images
        print("Using a total of ",len(self.img_list),"images")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,idx):
        img = Image.open(self.img_list[idx])
        if self.transform is not None:
            img = self.transform(img)
        if img.shape[0] == 1:
            img = np.repeat(img,3,0)
        return img

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
    parser.add_argument("--data_dir_path_1", type=str)
    parser.add_argument("--data_dir_path_2", type=str)    
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--debug",action="store_true")
    parser.add_argument("--img_size",type=int)
    parser.add_argument("--val_batch_size",type=int,default=50)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--result_fold_path",type=str,default="../results")
    parser.add_argument("--num_classes",type=int,default=15)
    args = parser.parse_args()

    if not os.path.exists(args.result_fold_path):
        os.makedirs(args.result_fold_path)

    cuda = torch.cuda.is_available()
    torch.set_grad_enabled(False)

    # Load model
    if args.model_path is None:
        model = models.densenet121(weights="IMAGENET1K_V1")
        if args.debug:
            print("Warning: no model path provided, using imagenet weights to debug")
        else:
            raise ValueError("No model path provided")
    else:
        model = models.densenet121(num_classes=args.num_classes)
        model.load_state_dict(torch.load(args.model_path))
    model.eval()
    if cuda:
        model = model.cuda()
    
    #add a hook on the last feature layer of the densenet121 model 
    #to get the feature vector
    vector_list = []
    def save_output(_,features,__):
        vector_list.append(features[0].cpu())
    model.classifier.register_forward_hook(save_output)

    stat_dic = {}

    for i,data_dir_path in enumerate([args.data_dir_path_1,args.data_dir_path_2]):

        dataset_name = get_dataset_name(data_dir_path)
        mu_path = args.result_fold_path+f"/mu_{dataset_name}.npy"
        sigma_path = args.result_fold_path+f"/std_{dataset_name}.npy"
        entropy_path = args.result_fold_path+f"/entropy_{dataset_name}.npy"

        pred_list = []

        if not os.path.exists(mu_path) or not os.path.exists(entropy_path):
            # Load image folder
            dataset = SynthImageFolder(data_dir_path,transform=transforms.Compose([
                                                                transforms.Resize(args.img_size),
                                                                transforms.CenterCrop(args.img_size),
                                                                transforms.ToTensor()]))
            
            dataloader = DataLoader(dataset,batch_size=args.val_batch_size,shuffle=False,num_workers=args.num_workers)

            for img in dataloader:
                if cuda:
                    img = img.cuda()
                pred_list.append(model(img).cpu())

            vectors = torch.cat(vector_list).numpy()
            vector_list = []

            mu = np.mean(vectors, axis=0)
            std = np.cov(vectors, rowvar=False)

            #Saves mu and std in npy files 
            np.save(mu_path,mu)
            np.save(sigma_path,std)

            preds = torch.cat(pred_list,dim=0)
            preds = torch.softmax(preds,dim=-1)
            preds = torch.clamp(preds,np.finfo(float).eps,1)
            entropy = (-preds*torch.log(preds)).sum(dim=1).mean(dim=0).numpy()
            np.save(entropy_path,entropy)

        else:
            mu = np.load(mu_path)
            std = np.load(sigma_path)
            entropy = np.load(entropy_path)

        stat_dic[dataset_name] = {"mu":mu,"std":std,"entropy":entropy}

    dataset1 = get_dataset_name(args.data_dir_path_1)
    dataset2 = get_dataset_name(args.data_dir_path_2)

    fid = calculate_frechet_distance(stat_dic[dataset1]["mu"],stat_dic[dataset1]["std"],stat_dic[dataset2]["mu"],stat_dic[dataset2]["std"])
    entropy1 = stat_dic[dataset1]["entropy"]
    entropy2 = stat_dic[dataset2]["entropy"]

    csv_path = args.result_fold_path+"/fid.csv"

    #if csv does not exists, create it with header 
    if not os.path.exists(csv_path):
        with open(csv_path,"w") as f:
            f.write("model_path,data_dir_path_1,data_dir_path_2,fid,entropy1,entropy2\n")

    with open(csv_path,"a") as f:
        f.write(f"{args.model_path},{dataset1},{dataset2},{fid},{entropy1},{entropy2}\n")

if __name__ == "__main__":
    main()
