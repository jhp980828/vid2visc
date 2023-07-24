import os
from os.path import join
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from MaskDataset import RandomAffine, transform_batch
from MaskAutoencoder import MaskAutoencoder
############################################################################################################
def train_and_test(model,
                   batch_size,
                   num_epochs,
                   optimizer,
                   loss_func,
                   train_data_path = "./processed_data/clf_dataset_Train_data.pt",
                   test_data_path = "./processed_data/clf_dataset_Test_data.pt",
                   device = "cuda:0" if torch.cuda.is_available() else "cpu",
                   save_dir = "./Models/AE_default.ckpt",
                   ):
    # Load Data
    all_train_data = torch.load (train_data_path)
    all_test_data = torch.load (test_data_path)
    print (f"Train set: {all_train_data.shape}")
    print (f"Test set: {all_test_data.shape}")
    
    # Dataloader
    train_loader = DataLoader (all_train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) # shuffled
    test_loader = DataLoader (all_test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    best_test_loss = torch.inf
    all_train_losses = []
    
    # Augmentation (trans,rot)
    # aug_transform = RandomAffine (p=0.5)
    # Train
    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        transform = RandomAffine(p=0.8, seed=epoch) ### transform augment
        print ("-"*100, f"Epoch {epoch}")
        model.train()

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            
            ### provide augmented batch after some epoch
            # if epoch >= 5:
            #     data = transform_batch(data, transform=transform)

            data = data.to(device) # block of fluid masks (B,1,12,H,W)
            out = model(data)
            loss = loss_func(out, data)
            all_train_losses.append (loss.item())
            loss.backward()
            optimizer.step()

            if i%5==0:
                print (f"Batch{i} Train Loss: {loss.item():1.5f}")

        # Test
        total_test_loss = 0.0
        model.eval()
        for i, data in enumerate (test_loader):
            with torch.no_grad():
                data = data.to(device)
                out = model(data)

                loss = loss_func(out,data)
                total_test_loss += loss.item()
        print (f"Test Loss: {total_test_loss:1.5f}")
        # Checkpoint
        if total_test_loss < best_test_loss:
            best_test_loss = total_test_loss

            checkpoint = {
                "MaskAutoencoder":model,
                "all_train_losses":all_train_losses,
                "total_test_loss":total_test_loss
                }
    
            torch.save (checkpoint, save_dir)
            print (f"Model saved after epoch {epoch}")
        print ()
    return None



############################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Learning Parameters
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--latent', type=int, default=256)
    
    # Others
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default="./exp")
    parser.add_argument('--save_name', type=str, default="autoencoder.ckpt")
    parser.add_argument('--mode', type=str, default="clf", help="clf or reg")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.makedirs (args.save_dir, exist_ok=True)

    ### data paths
    if args.mode == "clf":
        trainset_root = "../../clf_dataset_aug/masks/Train"
        testset_root = "../../clf_dataset_aug/masks/Test"
        TRAIN_PATH = "../processed_data/clf_dataset_aug_Train_data.pt"
        TEST_PATH = "../processed_data/clf_dataset_aug_Test_data.pt"
    elif args.mode == "reg":
        trainset_root = "../../reg_dataset_aug/masks/Train"
        testset_root = "../../reg_dataset_aug/masks/Test"
        TRAIN_PATH = "../processed_data/reg_dataset_aug_Train_data.pt"
        TEST_PATH = "../processed_data/reg_dataset_aug_Test_data.pt"
    else:
        print ("mode should be 'clf' or 'reg'")
        exit(1)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # num_train_vids = len(os.listdir(trainset_root))
    # --------------------------------------------------------------------------------
    ### Model
    ### NOTE: sample input
    sample_input = torch.rand (10, 1, 12, 40, 100) # b,c,t,h,w
    model = MaskAutoencoder (sample_input=sample_input, latent_dim=args.latent)
    optimizer = optim.Adam (model.parameters(), lr=args.lr, weight_decay=1e-5)
    model_save_path = join(args.save_dir, args.save_name)
    print (args.save_name)
    # --------------------------------------------------------------------------------
    train_and_test(
        model = model,
        batch_size = args.batch_size,
        num_epochs = args.epochs,
        optimizer = optimizer,
        loss_func = nn.MSELoss(), # reconstruction loss
        train_data_path = TRAIN_PATH,
        test_data_path = TEST_PATH,
        device = device,
        save_dir = model_save_path
        )
    
    config = {
        'name': args.save_dir,
        'mode': args.mode,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'latent_dim': args.latent,
        'seed': args.seed
    }
    config_dir = join (args.save_dir, "AEconfig.yaml")
    with open(config_dir, 'w') as file:
        documents = yaml.dump(config, file)


