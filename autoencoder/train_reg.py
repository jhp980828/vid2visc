import os
from os.path import join
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
# import torch.nn.functional as F
from tqdm import tqdm
import yaml

from MaskDataset import *
from utils import *


class ViscosityFCN (nn.Module):
    def __init__(self, AE_path):
        """
        Latent representations to viscosity
        """
        super(ViscosityFCN, self).__init__()
        # load encoder
        AE, saved_test_loss = load_saved_AE (AE_path)
        self.latent_dim = AE.latent_dim
        self.Encoder = nn.Sequential (
            AE.encoder,
            nn.Flatten(1,-1),
            AE.fcn_encode
            )

        self.fcn = nn.Sequential (
            nn.Linear (self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear (self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear (self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear (self.latent_dim, 1),
            )
        
    def forward (self, x):
        # x: mask data to encode with AE: (N,1,12,H,W)
        x = self.Encoder(x)
        # TODO;: PE
        # regression head
        x = self.fcn (x) # (N,1)
        return x



def trainFCN(model, train_loader, optimizer, loss_func, device, transform=None, loss_list=[]):
    ### TRAIN
    model = model.to(device)
    model.train()
    
    for i, (data,label) in enumerate(train_loader): # (N,1,12,30,80) data, one-hot label
        optimizer.zero_grad()
        
        if transform:
            data = transform_batch(data, transform)

        data = data.to(device)
        label = label.to(device) # viscosity label (one label per 12-frame data)
        label = label.reshape (-1,1)
        pred = model (data)

        loss = loss_func (pred, label)
        loss_list.append (loss.item())
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print (f"Batch {i}, Train loss: {loss.item():1.6f}")
    return loss_list


@torch.no_grad()
def testFCN (model, test_loader, loss_func):
    ### TEST
    model.eval()
    total_test_loss = 0.0
    for data,label in test_loader:
        data = data.to(device)
        label = label.reshape(-1,1).to(device)
        
        test_pred = model (data)
        test_loss = loss_func(test_pred, label)
        total_test_loss += test_loss.item()
    return total_test_loss


def load_saved_FCN (AE_path:str, FCN_path:str):
    model_root = os.path.dirname(AE_path)
    FCN_path = join(model_root, FCN_path)
    model = ViscosityFCN (AE_path=AE_path)
    FCN_statedict = torch.load (FCN_path)
    model.load_state_dict (FCN_statedict,)
    model.eval()
    return model


#########################################################################################################################################################
if __name__ == "__main__":
    ### Finetuning arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--AE_path', type=str, default="../out_dir/exp_reg_512/AE.ckpt")
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    
    ### Dataset
    parser.add_argument('--train_data', type=str, default="../processed_data/reg_dataset_aug_Train_data.pt")
    parser.add_argument('--train_labels', type=str, default="../processed_data/reg_dataset_aug_Train_label.pt")
    parser.add_argument('--test_data', type=str, default="../processed_data/reg_dataset_aug_Test_data.pt")
    parser.add_argument('--test_labels', type=str, default="../processed_data/reg_dataset_aug_Test_label.pt")

    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--save_name', type=str, default="FCNparams.ckpt")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device="cuda" if torch.cuda.is_available() else "cpu"
    save_dir = os.path.dirname(args.AE_path)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    ### Pretrained autoencoder and dataset
    AE, _ = load_saved_AE (args.AE_path)
    latent_dim = AE.latent_dim

    ### Mask features to be encoded by AE
    train_data = torch.load (args.train_data)
    train_labels = torch.load (args.train_labels)
    train_dataset = TensorDataset (train_data, train_labels)
    train_loader = DataLoader (train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    del train_data, train_labels, train_dataset

    test_data = torch.load (args.test_data)
    test_labels = torch.load (args.test_labels)
    test_dataset = TensorDataset (test_data, test_labels)
    test_loader = DataLoader (test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    del test_data, test_labels, test_dataset
    

    ### Regression head with pretrained encoder
    model = ViscosityFCN (AE_path=args.AE_path).to(device)
    
    loss_func = nn.MSELoss() # try MAE, or L1 if applicable
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ### Freeze pretrained portion
    if args.freeze_encoder:
        for param in model.Encoder.parameters():
            param.requires_grad = False
    else:
        for param in model.Encoder.parameters():
            param.requires_grad = True
    

    best_test_loss = torch.inf
    train_loss_list = []
    print (f"save_dir: {save_dir}")

    for epoch in range (args.epochs):
        print ("-"*100, f" Epoch {epoch}")
        transform = None # RandomAffine (p=0.8, seed=epoch)
        
        ### TRAIN LOOP
        train_loss_list = trainFCN (model = model,
                                    train_loader = train_loader,
                                    optimizer = optimizer,
                                    loss_func = loss_func,
                                    device = device,
                                    transform = transform, # rotation, translation
                                    loss_list = train_loss_list)
        
        ### TEST LOOP
        total_test_loss = testFCN (model = model,
                                   test_loader = test_loader,
                                   loss_func = loss_func)
        print (f"Test Loss: {total_test_loss:1.6f}")

        if total_test_loss < best_test_loss:
            best_test_loss = total_test_loss
            
            # save state dict
            torch.save (model.state_dict(), join(save_dir, args.save_name))
            print (f"\nFCN saved after epoch {epoch}")

    ### Save train loss plot
    plt.figure (figsize=(12,4))
    plt.plot (train_loss_list)
    plt.title ("Train Losses vs Iterations")
    plt.xlabel ("Iterations")
    plt.ylabel ("Train loss")
    plt.savefig(join(save_dir, args.save_name[:-5]+"_train_losses.jpg"))

    ### Save config.yaml
    config = {
        'name': join(save_dir,args.save_name),
        'base_encoder': args.AE_path,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'latent_dim': latent_dim,
        'seed': args.seed
    }
    
    modelname = args.save_name.split(".")[0] + "_config.yaml"
    config_dir = join (save_dir, modelname)
    with open(config_dir, 'w') as file:
        documents = yaml.dump(config, file)
