import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from os.path import join
from utils import loadVid
# ====================================================================================================================================================================================
### NOTE: Threshold remains the same for all videos.
# start_idx and stop_idx might have to be adjusted for videos ending with "_", which are shorter than 13s.
# Otherwise increasing the range of start/stop indices is an option with computational cost tradeoff.

# 20-40 for short videos
# 50-70 for 13s videos

def OpticalFlowTrim (video_path:str, start_idx=20, stop_idx=40, num_frames=150, threshold=3.0e6):
    """
    [Args]
    * video_path    : Video path string
    * start_idx     : First index in video frames to start motion calculation
    * stop_idx      : Last index
    * num_frames    : Length of the video
    """
    video = loadVid(video_path)
    D,H,W,C = video.shape
    motion = np.zeros(stop_idx-start_idx)
    assert stop_idx < D

    ### Farneback optical flow for motion
    for i in range(start_idx, stop_idx):
        prev_frame = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(video[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        ### store sum abs motions
        motion[i-start_idx] = np.sum(np.abs(flow))
    
    # threshold = 3.0e6
    trim_idx = np.where(motion < threshold)[0][0]
    
    # video that starts with same visual cue and trimmed to same num_frames
    video = video[start_idx+trim_idx : start_idx+trim_idx+num_frames]
    return video, motion, trim_idx


# ====================================================================================================================================================================================
def save_video_as_mp4(video, output_filename, fps=30):
    D, H, W, C = video.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (W, H))
    for i in range(D):
        frame = video[i]
        # Convert to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Frame', frame_rgb)
        
        # Write the frame
        out.write(frame)
        
        # Display the resulting frame    
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # After the loop release the capture
    cv2.destroyAllWindows() 
    # Release the video writer
    out.release()
    return


# ====================================================================================================================================================================================
def InteractiveTrim (video_path:str, start_idx=20, stop_idx=40, num_frames=150, threshold=3.0e6, output_filename="./tes4.mp4"):
    video = loadVid(video_path)
    D,H,W,C = video.shape
    saved = False
    while not saved:
        motion = np.zeros(stop_idx-start_idx)
        assert stop_idx < D, "stop_idx > D"

        fig,ax = plt.subplots (1,2, figsize=(16,5))
        ax[0].imshow (video[start_idx])
        ax[0].set_title ("start_idx: robot must be at left")
        ax[1].imshow (video[stop_idx])
        ax[1].set_title ("stop_idx: robot must be at right")
        plt.show ()

        resp = input ("Correct? (Y/N):   ")
        accepted_resp = ["Y", "y", "yes", "Yes", "YES"]
        if resp in accepted_resp:    
            plt.close ()
            ### Farneback optical flow for motion
            for i in range(start_idx, stop_idx):
                prev_frame = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
                next_frame = cv2.cvtColor(video[i + 1], cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                ### store sum abs motions
                motion[i-start_idx] = np.sum(np.abs(flow))
            
            # threshold = 3.0e6
            try:
                trim_idx = np.where(motion < threshold)[0][0]
                plt.figure()
                plt.plot (motion, "o--")
                plt.plot (trim_idx, motion[trim_idx], "ro")
                plt.title ("Framewise Optical Flow Motion")
                plt.show ()
                resp2 = input ("Is motion plot correct? (Y/N):   ")
            except:
                print ("motion array is empty")
                resp2 = "N"

            if resp2 in accepted_resp:
                # video that starts with same visual cue and trimmed to same num_frames
                processed_video = video[start_idx+trim_idx : start_idx+trim_idx+num_frames]
                save_video_as_mp4 (processed_video, output_filename=output_filename)
                saved = True
            else:
                print (f"previous indices: ({start_idx}, {stop_idx})")
                start_idx = int(input ("Input new start_idx:   "))
                stop_idx = int(input ("Input new stop_idx:   "))
                continue
        else:
            print (f"previous indices: ({start_idx}, {stop_idx})")
            start_idx = int(input ("Input new start_idx:   "))
            stop_idx = int(input ("Input new stop_idx:   "))
            continue
    return None


def lazy_main ():

    for i, path in enumerate (video_paths):
        out_filename = join(savedir,video_names[i])
        print ("-"*150)
        print (f"save_dir: {out_filename}")
        video, motion, trim_idx = OpticalFlowTrim(path, start_idx=start_idx, stop_idx=stop_idx, num_frames=num_frames, threshold=3.0e6)
        # print (trim_idx)
        
        if trim_idx > 0:
            save_video_as_mp4 (video, output_filename=out_filename)
        else:
            print (f"{path} unsaved")
            # with open("./unsaved_names.txt", "a") as file: # append
            #     file.write (str(path)+"\n")
    print ("-"*150)
    return

###########################################################################################################################################################################################
if __name__ == "__main__":

    root = "../../clf_aug_data/trim10s"
    savedir = join(os.path.dirname(root), "trim5s")
    video_names = os.listdir (root)
    video_names = [name for name in video_names if name.endswith("mp4")]
    video_names.sort()
    video_paths = [join(root,name) for name in video_names]

    start_idx = 10
    stop_idx = 20
    num_frames = 60

    # lazy_main ()
    
    
    for i, path in enumerate (video_paths):

        out_filename = join(savedir, video_names[i])
        print (f"save_dir: {out_filename}")
        try:
            InteractiveTrim (path, start_idx=start_idx, stop_idx=stop_idx, num_frames=num_frames, threshold=3.0e6,
                             output_filename=out_filename)
        except KeyboardInterrupt:
            print ("exit")
            exit()