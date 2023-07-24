import numpy as np
import torch
from torchvision.transforms import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from MaskDataset import *
import cv2
from torch.utils.data import DataLoader

# ==============================================================================================================================================================
def pickling (data, filename, load=False):
    """
    [Input]
    * data      : Data to save
    * filename  : Filename to be loaded or saved
    * load      : If set to false saves the data, else loads them.
    """
    if load:
        with open(filename, 'rb') as file:
            loaded_data = pickle.load(file)    
        return loaded_data
    else:
        with open(filename, 'wb') as file:
            pickle.dump (data, file)
        return None

# ==============================================================================================================================================================
def load_saved_AE (modelpath, device="cpu"):
    """
    [Input]
    * modelpath : Trained model path. ex) "./Models/AE/{name}"
    * device

    [Return]
    * model     : Loaded pretrained MaskAutoencoder
    """
    modeldict = torch.load (modelpath)
    model = modeldict["MaskAutoencoder"].to(device)
    test_acc = modeldict["total_test_loss"]
    return model, test_acc




# ==============================================================================================================================================================
def plot_train_loss (modelpath):
    """
    Plot saved train losses of AE or ViscFCN
    """
    model_dict = torch.load (modelpath)
    all_train_losses = model_dict["all_train_losses"] # list of floats
    plt.figure (figsize=(10,5))
    plt.plot (all_train_losses)
    plt.xlabel ("Batch")
    plt.ylabel ("Train Loss")
    plt.title ("Train Loss per Batch")
    plt.savefig (join(modelpath[:-5]+"_train_losses.jpg"))
    return

# ==============================================================================================================================================================
def plot_whatever(data, savefig=False):
    """
    Plot 12 frames (single data point)
    """
    fig,ax=plt.subplots(3,4,figsize=(18,4))
    if len(data.shape) == 3:
        for i,A in enumerate (ax.flatten()):
            A.imshow (data[i])
    elif len(data.shape) == 4:
        for i,A in enumerate (ax.flatten()):
            A.imshow (data[0][i])
    elif len(data.shape) == 5:
        try:
            data = data.detach().cpu().numpy()
        except:
            data = data.detach().numpy()
        for i,A in enumerate (ax.flatten()):
            A.imshow (data[0][0][i])

    if savefig:
        save_dir = input("Set filename without ext: ")
        save_dir += ".jpg"
        plt.savefig(save_dir)
    plt.show()
    return

# ==============================================================================================================================================================
def play_video(video):
    """
    [Input]
    video   : numpy array of shape (num_frames, H, W)
    """
    T, H, W = video.shape
    for t in range(T):
        frame = video[t, :, :]
        frame = cv2.normalize(frame, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frame = np.stack([frame]*3, axis=-1)
        cv2.imshow('Video', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return

# ==============================================================================================================================================================
def display_tensor(tensor, stop_at=None):
    tensor = tensor.cpu().detach()
    for i, frame in enumerate(tensor):
        frame = frame[0, 0, :, :]
        frame = frame.numpy()
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
        if stop_at is not None and i == stop_at:
            break
    cv2.destroyAllWindows()
    return

# ==============================================================================================================================================================
def loadVid (path):
    cap = cv2.VideoCapture(path)
    frames = []

    if cap.isOpened() == False:
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read() # (H,W,C) numpy
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    frames = np.stack(frames)
    return frames


# ==============================================================================================================================================================

def arr (tensor):
    return tensor.detach().cpu().numpy()


def encode_processed_data (autoencoder, dataset, batch_size=256, shuffle=False, num_workers=1):
    dataloader = DataLoader (dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    latents=[]
    autoencoder = autoencoder.cuda()
    autoencoder.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.cuda()
            tmp_latent_vector = autoencoder (data, encode=True)
            latents.append (tmp_latent_vector)
    latent_vectors = torch.cat (latents)
    return latent_vectors


def compare_recon (original_data, reconstruction, idx, cmap="gray", save_dir=None):
    idx_3d = (idx, 0, 0)
    fig,ax = plt.subplots (1, 2, figsize=(20,10))

    ax[0].imshow (original_data[idx_3d], cmap=cmap)
    # ax[0].set_title ("Original Data")
    ax[0].axis ("off")
    ax[1].imshow (reconstruction[idx_3d] > 0.5, cmap=cmap)
    # ax[1].set_title ("Reconstruction")
    ax[1].axis ("off")
    plt.tight_layout()

    if save_dir:
        name = f"compare_idx{idx}.jpg"
        name = join (save_dir, name)
        plt.savefig (name)
        plt.close ()
    else:
        plt.show ()
    return None


# @torch.no_grad()
# def get_latent_vectors(AE_path = "./out_dir/exp_reg_latent4/AE.ckpt",
#                        data_path = "./processed_data/reg_dataset_aug_Train_data.pt",
#                        device = "cpu"):    
#     ### load pretrained AE
#     AE, saved_test_loss = load_saved_AE (AE_path)
#     AE.eval()
#     AE = AE.to(device)

#     ### load mask features to be encoded by AE
#     features = torch.load (data_path)
#     print (data_path)
#     print (features.shape)

#     ### encode data to latent vectors
#     dataloader = DataLoader (features, batch_size=256, shuffle=False)
#     latent_vectors = []
#     for i, data in enumerate (dataloader):
#         latent_vector_batch = AE (data, encode=True)
#         latent_vectors.append (latent_vector_batch.detach().cpu())
#         print (f"Batch {i} encoded to {tuple(latent_vector_batch.shape)}")
#     latent_vectors = torch.cat(latent_vectors)
#     print (f"\nGenerated latent vectors: {latent_vectors.shape}")
#     return latent_vectors
