import os
from os.path import join
import torch
import numpy as np
import cv2
from fluid_segment import FCN_NetModel as FCN
from fluid_segment import CategoryDictionary as CatDic

### NOTE
# Segmentation model from RGB videos to fluid masks




def fluid_segmentation (InputVideo,
                        UseGPU=True,
                        FreezeBatchNormStatistics=False,
                        Trained_model_path=r"logs/trained_2023.torch"):
    """
    Model Ref: VectorLabPics
    XVID binary segmentation masks are saved to video directory.
    
    """

    OutVideoMain=InputVideo[:-4]+"_Filled.avi"
    MaskVideoMain=InputVideo[:-4]+"_Masks.avi"

    Net = FCN.Net(CatDic.CatNum)
    if UseGPU==True:
        print("USING GPU")
        Net.load_state_dict(torch.load(Trained_model_path))
    else:
        print("USING CPU")
        Net.load_state_dict(torch.load(Trained_model_path, map_location=torch.device('cpu')))
    
    cap = cv2.VideoCapture(InputVideo)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    MainVideoWriter=None
    MaskVideoWriter=None
    MaskMain = []

    while (cap.isOpened()):
        ret, Im = cap.read()
        if ret == False:
            break
        
        h,w,d = Im.shape
        r = np.max([h,w])

        if r > 840:
            fr = 840/r
            Im = cv2.resize(Im,(int(w*fr),int(h*fr)))
        h, w, d = Im.shape
        Imgs = np.expand_dims(Im,axis=0)
        
        if not (type(Im) is np.ndarray):
            continue

        #................................Make Prediction.............................................................................................................
        with torch.autograd.no_grad():
            OutProbDict,OutLbDict=Net.forward(Images=Imgs,TrainMode=False,UseGPU=UseGPU, FreezeBatchNormStatistics=FreezeBatchNormStatistics)
        #................................Make Prediction.............................................................................................................

        my=2
        mx=2
        OutMain = np.zeros(Im.shape, np.uint8) # 475 840 3
        nm = "Filled"

        y = 0
        x = 0
        OutMain[:h,:w]=Im
        
        Lb = OutLbDict[nm].data.cpu().numpy()[0].astype(np.uint8) # mask
        if Lb.mean()<0.001:
            continue

        if nm=='Ignore':
            continue

        # DISPLAYING SEGMENTED VIDEO ("Filled")
        ImOverlay1 = Im.copy()
        ImOverlay1[:, :, 1][Lb==1] = 0
        ImOverlay1[:, :, 0][Lb==1] = 255
        OutMain[h*y:h*(y+1), w*x:w*(x+1)] = ImOverlay1
        x+=1
        if x>=mx:
            x=0
            y+=1

        # cv2.imshow('Main Classes', OutMain)
        # cv2.waitKey(5)

        # SAVE "Filled"
        if MainVideoWriter is None:
            h, w, d = OutMain.shape
            MainVideoWriter = cv2.VideoWriter(OutVideoMain, fourcc, 30.0, (w, h))
        MainVideoWriter.write(OutMain)

        #-------------------------------------------------------------------------------------------------------------------------
        # FOR MASK VIDEO
        MaskMain.append (Lb)

    # SAVE Mask video
    MaskStack = np.stack(MaskMain)
    fourcc = cv2.VideoWriter_fourcc(*"XVID") # grayscale
    # h,w,d = OutMain.shape
    if MaskVideoWriter is None:
        MaskVideoWriter = cv2.VideoWriter (MaskVideoMain, fourcc, 30.0, (w,h), isColor=False)

    for i, mask in enumerate (MaskStack):
        mask = (mask * 255).astype(np.uint8)
        MaskVideoWriter.write (mask)
    #-----------------------------------------------------------------------------------------------------------------------------
    print("Finished")
    MainVideoWriter.release()
    MaskVideoWriter.release()
    cap.release()
    cv2.destroyAllWindows()
    return

############################################################################################################################################

def fluid_segment_ret (InputVideo,
                       UseGPU=True,
                       FreezeBatchNormStatistics=False,
                       Trained_model_path=r"logs/trained_2023.torch"):
    """
    Segmentation masks are RETURNED as numpy arrays for the autoencoder pipeline.
    
    """
    OutVideoMain=InputVideo[:-4]+"_Filled.avi"
    MaskVideoMain=InputVideo[:-4]+"_Masks.avi"

    Net = FCN.Net(CatDic.CatNum)
    if UseGPU==True:
        print("USING GPU")
        Net.load_state_dict(torch.load(Trained_model_path))
    else:
        print("USING CPU")
        Net.load_state_dict(torch.load(Trained_model_path, map_location=torch.device('cpu')))

    cap = cv2.VideoCapture(InputVideo)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    MainVideoWriter=None
    MaskVideoWriter=None
    MaskMain = []

    while (cap.isOpened()):
        ret, Im = cap.read()
        if ret == False:
            break
        
        h,w,d = Im.shape
        r = np.max([h,w])

        if r > 840:
            fr = 840/r
            Im = cv2.resize(Im,(int(w*fr),int(h*fr)))
        h, w, d = Im.shape
        Imgs = np.expand_dims(Im,axis=0)
        
        if not (type(Im) is np.ndarray):
            continue
        #................................Make Prediction.............................................................................................................
        with torch.autograd.no_grad():
            OutProbDict,OutLbDict=Net.forward(Images=Imgs,TrainMode=False,UseGPU=UseGPU, FreezeBatchNormStatistics=FreezeBatchNormStatistics)
        #................................Make Prediction.............................................................................................................
        my=2
        mx=2
        OutMain = np.zeros(Im.shape, np.uint8)
        nm = "Filled"

        y = 0
        x = 0
        OutMain[:h,:w]=Im
        
        Lb = OutLbDict[nm].data.cpu().numpy()[0].astype(np.uint8) # mask
        if Lb.mean()<0.001:
            continue

        if nm=='Ignore':
            continue

        # DISPLAYING SEGMENTED VIDEO ("Filled")
        ImOverlay1 = Im.copy()
        ImOverlay1[:, :, 1][Lb==1] = 0
        ImOverlay1[:, :, 0][Lb==1] = 255
        OutMain[h*y:h*(y+1), w*x:w*(x+1)] = ImOverlay1
        x+=1
        if x>=mx:
            x=0
            y+=1

        # SAVE "Filled"
        if MainVideoWriter is None:
            h, w, d = OutMain.shape
            MainVideoWriter = cv2.VideoWriter(OutVideoMain, fourcc, 30.0, (w, h))
        MainVideoWriter.write(OutMain)
        #-------------------------------------------------------------------------------------------------------------------------
        # FOR MASK VIDEO
        MaskMain.append (Lb)
    MaskStack = np.stack(MaskMain)
    return MaskStack

# ##################################################################################################################################################################

### NOTE: USE ./fluid_segment/colab_segment.ipynb


if __name__ == "__main__":
    video_name = "C:/Users/jhp98/OneDrive/Desktop/AE_final/saved_videos/vol80_01.mp4" # argparse


    InputVideo=video_name
    UseGPU=True
    FreezeBatchNormStatistics=False
    Trained_model_path="./trained_2023.torch"
    ##########################################################
    fluid_segmentation(InputVideo=InputVideo,
                       UseGPU=UseGPU,
                       FreezeBatchNormStatistics=FreezeBatchNormStatistics,
                       Trained_model_path=Trained_model_path)
