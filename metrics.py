
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

class SynthImageFolder():

    def __init__(self,root,transform=None):

        self.root = root
        self.transform = transform
        self.img_list = os.listdir(root)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,idx):
        img = Image.open(self.root+"/"+self.img_list[idx])
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
    parser.add_argument("--data_dir_1", type=str)
    parser.add_argument("--data_dir_2", type=str)    
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--debug",action="store_true")
    parser.add_argument("--img_size",type=int)
    parser.add_argument("--val_batch_size",type=int,default=50)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--result_fold_path",type=str,default="../results")

    args = parser.parse_args()

    if not os.path.exists(args.result_fold_path):
        os.makedirs(args.result_fold_path)

    cuda = torch.cuda.is_available()
    torch.set_grad_enabled(False)

    # Load model
    model = models.densenet121(weights="IMAGENET1K_V1")
    if args.model_path is None:
        if args.debug:
            print("Warning: no model path provided, using imagenet weights to debug")
        else:
            raise ValueError("No model path provided")
    else:
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
    for i,data_dir in enumerate([args.data_dir_1,args.data_dir_2]):

        if data_dir.endswith("/"):
            data_dir = data_dir[:-1]

        dataset_name = data_dir.split("/")[-1]
        mu_path = args.result_fold_path+f"/mu_{dataset_name}.npy"
        sigma_path = args.result_fold_path+f"/std_{dataset_name}.npy"

        print(mu_path,data_dir,dataset_name)

        if not os.path.exists(mu_path):
            # Load image folder
            dataset = SynthImageFolder(data_dir,transform=transforms.Compose([
                                                                transforms.Resize(args.img_size),
                                                                transforms.CenterCrop(args.img_size),
                                                                transforms.ToTensor()]))
            
            dataloader = DataLoader(dataset,batch_size=args.val_batch_size,shuffle=False,num_workers=args.num_workers)

            for img in dataloader:
                if cuda:
                    img = img.cuda()
                model(img)

            vectors = torch.cat(vector_list).detach().numpy()

            mu = np.mean(vectors, axis=0)
            std = np.cov(vectors, rowvar=False)

            #Saves mu and std in npy files 
            np.save(mu_path,mu)
            np.save(sigma_path,std)
        else:
            mu = np.load(mu_path)
            std = np.load(sigma_path)

        stat_dic[i] = {"mu":mu,"std":std,"dataset_name":dataset_name}

    fid = calculate_frechet_distance(stat_dic[0]["mu"],stat_dic[0]["std"],stat_dic[1]["mu"],stat_dic[1]["std"])

    csv_path = args.result_fold_path+"/fid.csv"

    #if csv does not exists, create it with header 
    if not os.path.exists(csv_path):
        with open(csv_path,"w") as f:
            f.write("model_path,data_dir_1,data_dir_2,fid\n")

    with open(csv_path,"a") as f:
        f.write(f"{args.model_path},{stat_dic[0]['dataset_name']},{stat_dic[1]['dataset_name']},{fid}\n")


if __name__ == "__main__":
    main()
