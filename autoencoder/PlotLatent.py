from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, dirname, basename
import os
import argparse
from utils import *

def plot_train_latent(
        model,
        train_data,
        train_labels,
        class_labels = ["coffee", "dish", "mango", "oil", "water"],
        reduction = "pca",
        mode = "train",
        task = "clf",
        save_dir = "./"
        ):
    ### NOTE: For both train/testset
    ### skip the labels too if skipping the data
    assert len(train_data) == len(train_labels), "Data and labels must have the same length"
    
    ### Encode up to 10000 data
    train_data_splits = []
    for i in range (5):
        train_data_split = train_data[i*2000:(i+1)*2000]
        train_data_splits.append (train_data_split)
    del train_data
    
    out_list=[]
    model.eval ()
    with torch.no_grad():
        for i, data in enumerate(train_data_splits):
            train_latent_vectors = model (data, encode=True)
            train_latent_vectors = train_latent_vectors.detach().cpu().numpy()
            out_list.append(train_latent_vectors)
        
    ### numpy
    train_labels = train_labels.detach().numpy()
    train_latent_vectors = np.concatenate(out_list) # all vectors
    latent = train_latent_vectors.shape[1]
    print (f"Encoded: {train_latent_vectors.shape}, {train_labels.shape}")
    
    ### dimensionality reduction (latent -> 2D)
    if reduction == "pca":
        dim_red = PCA (2)
    elif reduction == "tsne":
        dim_red = TSNE (2)
    else:
        print ("Input 'pca' or 'tsne' for reduction")
        pass

    ### plot and save
    plot_save_dir = join(save_dir, f"{latent}_{mode}_{reduction}.jpg")
    latent2D = dim_red.fit_transform (train_latent_vectors)
    df = pd.DataFrame(data = latent2D, columns = ["PC1", "PC2"])

    if task == "clf":
        df["label"] = [class_labels[int(i)] for i in train_labels]
    elif task == "reg":
        visc_dict = {
            "0 vol%": 1.005,
            "20 vol%": 1.769,
            "30 vol%": 2.501,    
            "40 vol%": 3.750,
            "50 vol%": 6.050,
            "60 vol%": 10.96,
            "70 vol%": 22.94,
            "90 vol%": 234.6
            }
        visc_rev_dict = {v: k for k, v in visc_dict.items()}
        def closest_label(val):
            return min(visc_dict.items(), key=lambda x: abs(x[1] - val))[0]
        df["label"] = [closest_label(float(i)) for i in train_labels]
    else:
        exit()

    plt.figure(figsize=(10,10))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="label", palette="deep", s=20)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.title("2D Visualization of the Latent Representations", fontsize=16)
    plt.savefig(plot_save_dir)
    plt.close ()
    return None

# ==============================================================================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default="../AE.ckpt")
    parser.add_argument('--task', type=str, default="clf", help="clf or reg")
    args = parser.parse_args()
    modelpath = args.modelpath

    # ------------------------------------------------------------------------------------------------------------
    ### Dataset
    if args.task == "clf":
        train_data_path = "../processed_data/clf_dataset_aug_Train_data.pt"
        train_labels_path = "../processed_data/clf_dataset_aug_Train_label.pt"
        test_data_path = "../processed_data/clf_dataset_aug_Test_data.pt"
        test_labels_path = "../processed_data/clf_dataset_aug_Test_label.pt"
        class_labels = ["Coffee", "Dish Soap", "Mango Juice", "Oil", "Water"]
    elif args.task == "reg":
        train_data_path = "../processed_data/reg_dataset_aug_Train_data.pt"
        train_labels_path = "../processed_data/reg_dataset_aug_Train_label.pt"
        test_data_path = "../processed_data/reg_dataset_aug_Test_data.pt"
        test_labels_path = "../processed_data/reg_dataset_aug_Test_label.pt"
        class_labels = None
    else:
        exit()

    train_data = torch.load (train_data_path)
    train_labels = torch.load (train_labels_path)
    train_skip=3
    train_data_skip = train_data [::train_skip]
    train_labels_skip = train_labels [::train_skip]
    del train_data, train_labels

    test_data = torch.load (test_data_path)
    test_labels = torch.load (test_labels_path)
    
    # ------------------------------------------------------------------------------------------------------------
    ### train loss
    modelname = modelpath.split ("/")[-2]
    model, saved_loss = load_saved_AE (modelpath)
    print (modelname, saved_loss)
    plot_train_loss (modelpath)

    ### train latent rep (pca, tsne)
    plot_train_latent (model=model,
                       train_data=train_data_skip,
                       train_labels=train_labels_skip,
                       save_dir=dirname(modelpath),
                       mode = "train",
                       task = args.task,
                       class_labels = class_labels,
                       reduction="pca")


    plot_train_latent (model=model,
                       train_data=train_data_skip,
                       train_labels=train_labels_skip,
                       save_dir=dirname(modelpath),
                       mode = "train",
                       task = args.task,
                       class_labels = class_labels,
                       reduction="tsne")
    del train_data_skip, train_labels_skip

    ### testset
    plot_train_latent (model=model,
                       train_data=test_data,
                       train_labels=test_labels,
                       save_dir=dirname(modelpath),
                       mode = "test",
                       task = args.task,
                       class_labels = class_labels,
                       reduction="pca")


    plot_train_latent (model=model,
                       train_data=test_data,
                       train_labels=test_labels,
                       save_dir=dirname(modelpath),
                       mode = "test",
                       task = args.task,
                       class_labels = class_labels,
                       reduction="tsne")