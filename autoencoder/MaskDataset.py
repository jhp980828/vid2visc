import os
from os.path import join
import cv2
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms
from torchvision.transforms import transforms
from skimage.transform import resize


# ==============================================================================================================================================================
def loadMask(path):
    """
    [Input]
    * path: str
    
    [Return]
    * frames: numpy array of shape (num_frames, H, W); binary {0,1}, grayscale
    """
    cap = cv2.VideoCapture(path)
    frames = []
    if cap.isOpened()== False:
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read() # (H,W,C) numpy
        if ret:
            frame = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY) # (H,W)
            _, binary_frame = cv2.threshold(frame, 0, 1, cv2.THRESH_BINARY) # (H,W) in {0,1}
            frames.append(binary_frame)
        else:
            break
    cap.release()
    frames = np.stack(frames)
    if "clf" in path:
        frames = frames[:150]
    return frames


# ==============================================================================================================================================================
# sliding windows: [0:t], [1,t+1], [2,t+2], ...
def frames_to_data(frames, num_timesteps = 12):
    """
    [Input]
    * frames        : torch (1, num_frames, h, w)
    * num_timesteps : Number of frames for each data point (>12)

    [Return]
    * sliding_windows  : torch (num_data, 1, num_timesteps, h, w), num_data = num_frames - num_timesteps

    """
    _, num_frames, h, w = frames.shape
    num_data = num_frames-num_timesteps # one video is this many points (~260)

    sliding_windows = torch.zeros (num_data, 1, num_timesteps, h, w)
    for i in range (num_data):
        sliding_window = frames[:, i:(i+num_timesteps), :, :]
        sliding_windows[i] = sliding_window
    return sliding_windows
    

# ==============================================================================================================================================================
# 
# clf_dataset: crop_range=[65, 195, 80, 120]
# reg_dataset: crop_range=[80, 200, 85, 125]
class ResizeVideo:
    def __init__(self, resize_width=300, crop_range=[80, 200, 85, 125]):
        self.resize_width = resize_width
        self.crop_range = crop_range

    def __call__(self, video):
        """
        [Input]
        * video : (num_frames,H,W) numpy array of mask frames

        [Return]
        * cropped_video   : (num_frames, 40, 120) numpy array
        """
        # resize
        num_frames, h, w = video.shape
        if w > h:
            ow = self.resize_width
            oh = int(self.resize_width * h / w)
        else:
            oh = self.resize_width # longer side
            ow = int(self.resize_width * w / h)
        resized_video = resize(video, (num_frames, oh, ow), order=0) # nearest

        # crop
        xmin, xmax, ymin, ymax = self.crop_range # unpack crop range
        cropped_video = resized_video [:, ymin:ymax, xmin:xmax]
        return cropped_video


class RandomAffine:
    """
    - Get random transformation in reasonable range of (x,y,deg) values
    - Augmentation will cover the shaking of camera & desk, and slipping from gripper
    - Input seed=epoch to apply different augmentation every epoch
    
    """
    def __init__(self, p=0.8, seed=0):
        self.probability = p
        self.seed = seed

    def __call__(self, video):
        if np.random.random() < self.probability:
            # np.random.seed(self.seed) ############### comment out this line - Abe
            tx = np.random.randint (-2, 3)
            ty = np.random.randint (-2, 3)
            rot = np.random.uniform(-2, 3)
            augmented_video = torchvision.transforms.functional.affine(video, angle=rot, translate=(tx, ty), scale=1.0, shear=1) # torchvision
            return augmented_video
        else:
            return video

        
def transform_batch (batch_data, transform):
    """
    Function for transforming a batch using RandomAffine
    """
    batch_data = batch_data.clone()
    squeezed_batch_data = batch_data.squeeze(1) # squeeze channel dim (t,h,w)
    transformed = []
    for i, data3D in enumerate(squeezed_batch_data):
        data3D = transform(data3D)
        data3D = data3D.unsqueeze(0) # create num_data dim (1,t,h,w)
        transformed.append (data3D)
    transformed = torch.cat(transformed) # stack to (b,t,h,w)
    transformed = transformed.unsqueeze(1) # unsqueeze channel dim (b,c,t,h,w)
    return transformed



def default_transform_reg ():
    """used on reg_dataset_aug"""
    transform = transforms.Compose ([
        ResizeVideo(resize_width=256,crop_range=[60,160,70,110]),
        torch.tensor,
        # RandomAffine(p=0.8)
        ])
    return transform

def default_transform_clf ():
    """used on clf_dataset_aug"""
    transform = transforms.Compose ([
        ResizeVideo(resize_width=256,crop_range=[70,170,80,120]),
        torch.tensor,
        # RandomAffine(p=0.8)
        ])
    return transform

def trim_masks (video_frames, filename):
    if "vol0" in filename:
        video_frames[:, :, 502:] = 0.0 # lid
        video_frames[:, 0:240, 370:420] = 0.0 # robot arm
    elif "vol40" in filename:
        video_frames[:, :, 500:] = 0.0
        video_frames[:, 0:278, 375:420] = 0.0
    elif "vol60" in filename:
        video_frames[:, :, 505:] = 0.0
        video_frames[:, 0:285, 375:420] = 0.0
    elif "vol80" in filename:
        video_frames[:, :, 512:] = 0.0
        video_frames[:, 0:285, 375:425] = 0.0
    elif "vol100" in filename:
        video_frames[:, :, 512:] = 0.0
        video_frames[:, 0:278, 377:420] = 0.0
    elif "vol120" in filename:
        video_frames[:, :245, :] = 0.0
        video_frames[:, :, 512:] = 0.0
    elif "vol140" in filename:
        video_frames[:, :245, :] = 0.0
        video_frames[:, :, 505:] = 0.0
    elif "vol180" in filename:
        video_frames[:, :245, :] = 0.0
        video_frames[:, :, 518:] = 0.0
    elif "coffee" in filename:
        video_frames[:, :270, :] = 0.0
        video_frames[:, :, 532:] = 0.0
    elif "dish" in filename:
        video_frames[:, :270, :] = 0.0
        video_frames[:, :, 540:] = 0.0
    elif "mango" in filename:
        video_frames[:, :270, :] = 0.0
        video_frames[:, :, 537:] = 0.0
    elif "oil" in filename:
        video_frames[:, :270, :] = 0.0
        video_frames[:, :, 540:] = 0.0
    elif "water" in filename:
        video_frames[:, :270, :] = 0.0
        video_frames[:, :, 544:] = 0.0
    else:
        print ("not trimmed")
    return video_frames
# ==============================================================================================================================================================
class MaskDataset(Dataset):
    """
    [Args]
    * root            : Mask video root (./{dataset}/masks/{Train or Test})
    * transform       : Resize and crop
    * num_timesteps   : Temporal dimension of each data point (>12)
    
    [Items]
    * video_frames    : 5D tensor data (N,C,T,H,W) extracted from segmented video (sliding windows)
    * filename        : Filename that can be mapped to label

    """
    def __init__(self, root="./clf_dataset/masks/Test", transform=default_transform_clf(), num_timesteps=12):
        self.root = root
        self.transform = transform
        self.num_timesteps = num_timesteps
        self.mask_names = os.listdir (root)
        self.mask_names.sort() # alphabetized
        self.mask_dir = [join (root, mask_name) for mask_name in self.mask_names]
        self.num_videos = len(self.mask_dir)
        
    def __len__ (self):
        return self.num_videos
        
    def __getitem__(self, index):
        ### load and get name (label)
        video_frames = loadMask (self.mask_dir[index]) # (d,h,w) numpy
        label = self.get_filename(index) # map to label

        video_frames = trim_masks (video_frames, label)
        if self.transform:
            video_frames = self.transform (video_frames)
        d,h,w = video_frames.shape # transformed
        video_frames = video_frames.unsqueeze(0)
        video_frames = frames_to_data (video_frames, num_timesteps=self.num_timesteps)
        return video_frames, label
    
    def get_filename(self, index):
        video_filename = self.mask_names[index]
        return video_filename



class MaskInferenceDataset(Dataset):
    """
    Inference without sliding window. (D//d datapoints per video)
    """
    def __init__(self, root="./clf_dataset/masks/Test", transform=default_transform_reg(), num_timesteps=12):
        self.root = root
        self.transform = transform
        self.num_timesteps = num_timesteps
        self.mask_names = os.listdir (root)
        self.mask_names.sort() # alphabetized
        self.mask_dir = [join (root, mask_name) for mask_name in self.mask_names]
        self.num_videos = len(self.mask_dir)
        
    def __len__ (self):
        return self.num_videos
        
    def __getitem__(self, index):
        ### load and get name (label)
        video_frames = loadMask (self.mask_dir[index]) # (d,h,w) numpy
        label = self.get_filename(index) # map to label

        video_frames = trim_masks (video_frames, label)
        if self.transform:
            video_frames = self.transform (video_frames)
        d,h,w = video_frames.shape # transformed
        
        video_frames = video_frames.unsqueeze(0)
        video_frames = frames_to_data (video_frames, self.num_timesteps)

        video_frames = video_frames[::self.num_timesteps] # inference datapts
        video_frames = video_frames.float()
        return video_frames, label
    
    def get_filename(self, index):
        video_filename = self.mask_names[index]
        return video_filename

# ==============================================================================================================================================================
def concat_video_dataset (root, transform=default_transform_clf(), save_dir="./", num_timesteps=12):
    """
    Data processing with one-hot encoded labels

    [Input]
    * root        : "./oscil_dataset2/masks/Train" or "./oscil_dataset2/masks/Test"
    * save_dir    : Path to save 'data.pt' and 'label.pt'
    
    [Saved]
    * Data tensors as pt file
    * Label tensors as pt file
    """
    if root.endswith ("/"):
        root = root[:-1]
    sp = root.split("/")
    mode = sp[-1]       # str: "Train" or "Test"
    dataroot = sp[-3]    # str: dataset name
    
    # if no save dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # if there is a non empty save dir
    elif os.path.isdir(save_dir) and len(os.listdir(save_dir)) > 0 and mode=="Train":
        cont = input(f"{len(os.listdir(save_dir))} files already present in save_dir. Overwrite? (Y/N): ")
        if cont != "Y":
            print ("Terminated")
            exit(1)

    # if there is an empty save dir
    else:
        pass

    category_dict = {
        "coffee": 0,
        "dish": 1,
        "mango": 2,
        "oil": 3,
        # "syrup": 4,
        "water": 4
        }
    class_labels = [cls for cls in category_dict.keys()] # ["Water", "Milk", etc.]

    print (f"Dataset root: {root}")
    print (f"Building {mode} Set")
    
    ### MaskDataset
    dataset = MaskDataset (transform=transform, root=root, num_timesteps=num_timesteps)
    labels_list = []
    data_list = []
    for i, (data, name) in enumerate (dataset):
        print (f"{i} - {name} processed into {tuple(data.shape)}")
        num_data = data.shape[0]
        
        ### Store Features/One-hot labels
        data_list.append (data)
        for cls in class_labels:
            if cls in name:
                one_hot = int (category_dict[cls])
                labels = torch.ones(num_data) * one_hot
                labels_list.append (labels)
    data_list = torch.cat(data_list)
    labels_list = torch.cat(labels_list)
    
    ### Save
    torch.save (data_list, join(save_dir, f"{dataroot}_{mode}_data.pt"))
    torch.save (labels_list, join(save_dir, f"{dataroot}_{mode}_label.pt"))
    print ("\nConcatenated Dataset: ")
    print (f"Data: {tuple(data_list.shape)}")
    print (f"Labels: {tuple(labels_list.shape)}")
    print ("Saved")
    return None


def concat_video_dataset2 (root, transform=default_transform_clf(), save_dir="./", num_timesteps=12):
    """
    reg_dataset
    """
    if root.endswith ("/"):
        root = root[:-1]
    sp = root.split("/")
    mode = os.path.basename (root)  # str: "Train" or "Test"
    dataroot = sp[-3]    # str: dataset name; ex) "../../reg_aug_dataset/masks/Train"
    
    # if no save dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # if there is a non empty save dir
    elif os.path.isdir(save_dir) and len(os.listdir(save_dir)) > 0 and mode=="Train":
        cont = input(f"{len(os.listdir(save_dir))} files already present in save_dir. Overwrite? (Y/N): ")
        if cont != "Y":
            print ("Terminated")
            exit(1)

    # if there is an empty save dir
    else:
        pass
    
    # keys  : glycerol volume in 200ml solution
    # values: viscosity in cp
    category_dict = {
        "vol0": 1.005,     # pure water
        "vol40": 1.769,
        "vol60": 2.501,    
        "vol80": 3.750,
        "vol100": 6.050,
        "vol120": 10.96,
        "vol140": 22.94,
        "vol180": 234.6
        }
    class_labels = [cls for cls in category_dict.keys()]

    print (f"Dataset root: {root}")
    print (f"Building {mode} Set")
    
    ### MaskDataset
    dataset = MaskDataset (transform=transform, root=root, num_timesteps=num_timesteps)
    labels_list = []
    data_list = []
    for i, (data, name) in enumerate (dataset):
        print (f"{i} - {name} processed into {tuple(data.shape)}")
        num_data = data.shape[0]
        
        ### Store Features/One-hot labels
        data_list.append (data)
        for cls in class_labels:
            if cls in name:
                if "clf" in args.root:
                    one_hot = int (category_dict[cls])
                else:
                    one_hot = category_dict[cls]
                labels = torch.ones(num_data) * one_hot
                labels_list.append (labels)
    data_list = torch.cat(data_list)
    labels_list = torch.cat(labels_list)
    
    ### Save
    torch.save (data_list, join(save_dir, f"{dataroot}_{mode}_data.pt"))
    torch.save (labels_list, join(save_dir, f"{dataroot}_{mode}_label.pt"))
    print ("\nConcatenated Dataset: ")
    print (f"Data: {tuple(data_list.shape)}")
    print (f"Labels: {tuple(labels_list.shape)}")
    print ("Saved")
    return None


# ------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="../../reg_dataset_aug")
    parser.add_argument('--save_dir', type=str, default="../processed_data")
    parser.add_argument('--num_timesteps', type=int, default=12)
    args = parser.parse_args()
    
    ### convert a video folder to single tensor dataset
    if "clf" in args.root:
        concat_video_dataset (root=args.root,
                              transform=default_transform_clf(),
                              save_dir=args.save_dir,
                              num_timesteps=args.num_timesteps)
        
    elif "reg" in args.root:
        concat_video_dataset2 (root=args.root,
                               transform=default_transform_reg(),
                               save_dir=args.save_dir,
                               num_timesteps=args.num_timesteps)


