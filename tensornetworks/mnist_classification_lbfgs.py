import argparse
import datetime
import copy
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import utils
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default="mnist", type=str, help='dataset to use')
parser.add_argument('--penalty', default="l2", type=str, help='penalty for logistic regression')
parser.add_argument('--num_train_samples', default=1000, type=int, help='number of training samples')
parser.add_argument(
            '--target_size', 
            default=None, type=int,
            help = "target size of the coarse graining, using Wave's fractional coarse-graining scheme")
parser.add_argument(
            '--n_pca_components_kept',
            default=None, type=float, 
            help = "percentage of pca components to keep")
parser.add_argument(
            '--is_high_signal_to_noise',
            default="False", type=str,
            help = "whether the signal to noise ratio is high (i.e. retain high variance components)")
parser.add_argument(
            '--fileprefix', 
            default="",
            type=str, 
            action='store')
parser.add_argument(
            '--save_dir', 
            default="/scratch/gpfs/qanguyen/imagenet_info",
            type=str, 
            action='store')

args = parser.parse_args()

class RandomFeaturesMNIST(datasets.MNIST):
    def __init__(self, root = "./data",  
                 train = True,
                 transform = None,
                 upsample = False,
                 target_size = None,
                ):
             
         

        if target_size is not None:
            self.target_size = target_size
            self.transform_matrix = self.get_transformation_matrix(target_size, 28)
            self.retransform_matrix = self.get_transformation_matrix(28, target_size)
            
        self.upsample = upsample
         
        super(RandomFeaturesMNIST, self).__init__(root, train=train, transform=transform, download=True)
         
    def get_transformation_matrix(self, target_size, full_shape):
        # List of coarse-grained coordinates
        x1,y1=torch.meshgrid(torch.arange(target_size),torch.arange(target_size))
        x2,y2 = x1+1,y1+1
        x1,y1,x2,y2 = x1/target_size,y1/target_size,x2/target_size,y2/target_size
        r_prime = torch.vstack([x1.flatten(),x2.flatten(),y1.flatten(),y2.flatten()]).T 

        # List of fine-grained coordinates
        m1,n1=torch.meshgrid(torch.arange(full_shape),torch.arange(full_shape))
        m2,n2 = m1+1,n1+1
        m1,n1,m2,n2 = m1 / full_shape,n1 /full_shape,m2 / full_shape,n2 /full_shape 
        r = torch.vstack([m1.flatten(),m2.flatten(),n1.flatten(),n2.flatten()]).T 

        minrprimex1x2 = torch.minimum(r_prime[:,0], r_prime[:,1])
        minrx1x2 = torch.minimum(r[:,0], r[:,1])
        maxrprimex1x2 = torch.maximum(r_prime[:,0], r_prime[:,1])
        maxrx1x2 = torch.maximum(r[:,0], r[:,1])

        minrprimey1y2 = torch.minimum(r_prime[:,2], r_prime[:,3])
        minry1y2 = torch.minimum(r[:,2], r[:,3])
        maxrprimey1y2 = torch.maximum(r_prime[:,2], r_prime[:,3])
        maxry1y2 = torch.maximum(r[:,2], r[:,3])

        x1 = torch.maximum(minrprimex1x2.unsqueeze(0),minrx1x2.unsqueeze(1))
        x2 = torch.minimum(maxrprimex1x2.unsqueeze(0),maxrx1x2.unsqueeze(1)) 
        delta_x = torch.clamp(x2-x1,min=0)
        y1 = torch.maximum(minrprimey1y2.unsqueeze(0),minry1y2.unsqueeze(1))
        y2 = torch.minimum(maxrprimey1y2.unsqueeze(0),maxry1y2.unsqueeze(1)) 
        delta_y = torch.clamp(y2-y1,min=0)
        return delta_x * delta_y
        

    def __getitem__(self, index: int):
        sample, target = super(RandomFeaturesMNIST, self).__getitem__(index)
        if self.target_size == 28:
            return sample.flatten().numpy(), target
        
        if self.target_size is not None:
            sample = torch.matmul(sample.view(1, -1), self.transform_matrix)
            
            sample = sample.view(1, self.target_size, self.target_size) * self.target_size * self.target_size
            if self.upsample:
                sample = torch.matmul (sample.view(1, -1), self.retransform_matrix)
                sample = sample.view(1, 28, 28) * 28 * 28

        return sample.flatten().numpy(), target 
class RandomFeaturesFashionMNIST(datasets.FashionMNIST):
    def __init__(self, root = "./data",  
                 train = True,
                 transform = None,
                 upsample = False,
                 target_size = None,
                ):
        
         

        if target_size is not None:
            self.target_size = target_size
            self.transform_matrix = self.get_transformation_matrix(target_size, 28)
            self.retransform_matrix = self.get_transformation_matrix(28, target_size)
            
        self.upsample = upsample
         
        super(RandomFeaturesFashionMNIST, self).__init__(root, train=train, transform=transform, download=True)
         
    def get_transformation_matrix(self, target_size, full_shape):
        # List of coarse-grained coordinates
        x1,y1=torch.meshgrid(torch.arange(target_size),torch.arange(target_size))
        x2,y2 = x1+1,y1+1
        x1,y1,x2,y2 = x1/target_size,y1/target_size,x2/target_size,y2/target_size
        r_prime = torch.vstack([x1.flatten(),x2.flatten(),y1.flatten(),y2.flatten()]).T 

        # List of fine-grained coordinates
        m1,n1=torch.meshgrid(torch.arange(full_shape),torch.arange(full_shape))
        m2,n2 = m1+1,n1+1
        m1,n1,m2,n2 = m1 / full_shape,n1 /full_shape,m2 / full_shape,n2 /full_shape 
        r = torch.vstack([m1.flatten(),m2.flatten(),n1.flatten(),n2.flatten()]).T 

        minrprimex1x2 = torch.minimum(r_prime[:,0], r_prime[:,1])
        minrx1x2 = torch.minimum(r[:,0], r[:,1])
        maxrprimex1x2 = torch.maximum(r_prime[:,0], r_prime[:,1])
        maxrx1x2 = torch.maximum(r[:,0], r[:,1])

        minrprimey1y2 = torch.minimum(r_prime[:,2], r_prime[:,3])
        minry1y2 = torch.minimum(r[:,2], r[:,3])
        maxrprimey1y2 = torch.maximum(r_prime[:,2], r_prime[:,3])
        maxry1y2 = torch.maximum(r[:,2], r[:,3])

        x1 = torch.maximum(minrprimex1x2.unsqueeze(0),minrx1x2.unsqueeze(1))
        x2 = torch.minimum(maxrprimex1x2.unsqueeze(0),maxrx1x2.unsqueeze(1)) 
        delta_x = torch.clamp(x2-x1,min=0)
        y1 = torch.maximum(minrprimey1y2.unsqueeze(0),minry1y2.unsqueeze(1))
        y2 = torch.minimum(maxrprimey1y2.unsqueeze(0),maxry1y2.unsqueeze(1)) 
        delta_y = torch.clamp(y2-y1,min=0)
        return delta_x * delta_y
        

    def __getitem__(self, index: int):
        sample, target = super(RandomFeaturesFashionMNIST, self).__getitem__(index)
        # prev_sample = sample.clone()
        if self.target_size == 28:
            return sample.flatten().numpy(), target
        if self.target_size is not None:
            sample = torch.matmul(sample.view(1, -1), self.transform_matrix)
            
            sample = sample.view(1, self.target_size, self.target_size) * self.target_size * self.target_size
            if self.upsample:
                sample = torch.matmul (sample.view(1, -1), self.retransform_matrix)
                sample = sample.view(1, 28, 28) * 28 * 28

        return sample.flatten().numpy(), target 

transform = transforms.Compose([
            transforms.ToTensor(),
        ])
if args.data == "mnist":
    train_dataset = RandomFeaturesMNIST(root = "./data", 
                                    train = True,
                                    transform = transform,
                                    target_size = args.target_size,
                                    upsample = True
    )         
    test_dataset = RandomFeaturesMNIST(root = "./data", 
                                    train = False,
                                    transform = transform,
                                    target_size = args.target_size,
                                    upsample = True
    )
elif args.data == "fashionmnist":
    train_dataset = RandomFeaturesFashionMNIST(root = "./data", 
                                    train = True,
                                    transform = transform,
                                    target_size = args.target_size,
                                    upsample = True
    )   
        
    test_dataset = RandomFeaturesFashionMNIST(root = "./data", 
                                    train = False,
                                    transform = transform,
                                    target_size = args.target_size,
                                    upsample = True
    )

for seed in [0, 5, 10, 15, 20]:
    rng = np.random.default_rng(seed)
    num_train = len(train_dataset)
    train_idx = rng.integers(low=0, high=num_train, size=args.num_train_samples)

    X, y = zip(*[train_dataset[i] for i in range(num_train)]) # get all the training samples
    X = np.array(X)
    y = np.array(y) 
    
    if args.n_pca_components_kept is not None:
        print ("X.shape, y.shape", X.shape, y.shape, X[0])
        assert args.target_size == 28
        pca = PCA(n_components=None)
        pca.fit(X)
        Xpca = pca.transform(X) 
        if args.is_high_signal_to_noise != "False":
            args.n_pca_components_kept = 1-args.n_pca_components_kept
        n_pca_components_kept = int(args.n_pca_components_kept * Xpca.shape[1])
        assert n_pca_components_kept >= 0, f"n_pca_components_kept must be greater than 0, got {n_pca_components_kept}"
        high_signal, low_signal = copy.deepcopy(Xpca[:,:n_pca_components_kept]), copy.deepcopy(Xpca[:,n_pca_components_kept:])
        print ("high_signal.shape, low_signal.shape", high_signal.shape, low_signal.shape)
        if args.is_high_signal_to_noise == "False":
            # high_signal_shuffle = np.random.default_rng(seed).permutation(high_signal.shape[1])
            # high_signal = np.take(high_signal, high_signal_shuffle, axis=1) # shuffle the high signal
            np.random.shuffle(high_signal.T)
        else:
            # low_signal_shuffle = np.random.default_rng(seed).permutation(low_signal.shape[1])
            # low_signal = np.take(low_signal, low_signal_shuffle, axis=1) # shuffle the low signal
            np.random.shuffle(low_signal.T) 
        Xpca = np.hstack([high_signal, low_signal])
            
        X = pca.inverse_transform(Xpca)

    # fit on subset of the training data
    Xtrain = X[train_idx]
    ytrain = y[train_idx]
    print ("fitting Xtrain.shape, ytrain.shape", Xtrain.shape, ytrain.shape)
    clf = LogisticRegression(max_iter=1e12, penalty=args.penalty,solver='lbfgs',verbose=1).fit(Xtrain, ytrain)
    s = clf.score(Xtrain, ytrain)
    train_loss = log_loss(ytrain, clf.predict_proba(Xtrain))

    print ("score", s, "train_loss", train_loss, Xtrain.shape, ytrain.shape)

    Xtest, ytest = zip(*[test_dataset[i] for i in range(len(test_dataset))]) 
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)
    if args.n_pca_components_kept is not None:
        print ("Xtest.shape, ytest.shape", Xtest.shape, ytest.shape)
        Xtestpca = pca.transform(Xtest)
        high_signal, low_signal = copy.deepcopy(Xtestpca[:,:n_pca_components_kept]), copy.deepcopy(Xtestpca[:,n_pca_components_kept:])

        print ("high_signal.shape, low_signal.shape", high_signal.shape, low_signal.shape)
        if args.is_high_signal_to_noise == "False":
            # high_signal = np.take(high_signal, high_signal_shuffle, axis=1) # shuffle the high signal
            np.random.shuffle(high_signal.T)
        else:
            # low_signal = np.take(low_signal, low_signal_shuffle, axis=1) # shuffle the low signal
            np.random.shuffle(low_signal.T)
        Xtestpca = np.hstack([high_signal, low_signal])
        Xtest = pca.inverse_transform(Xtestpca)
    s_test = clf.score(Xtest, ytest)
    test_loss = log_loss(ytest, clf.predict_proba(Xtest))
    print ("score", s, s_test, test_loss, X.shape, y.shape)
    record = {"seed": seed, 
                "train_loss": train_loss, "test_loss": test_loss, 
                "train_score": s, "test_score": s_test, 
                "num_train_samples": args.num_train_samples, 
                "n_pca_components_kept": n_pca_components_kept,
                "target_size": args.target_size, "args":args}
    args.exp_name = f"{args.fileprefix}" \
                + f"_rep_{datetime.datetime.now().timestamp()}.pth.tar"
    print(f"saved to {args.save_dir}/{args.exp_name}" )

    utils.save_checkpoint(record, save_dir = args.save_dir, filename = args.exp_name) 