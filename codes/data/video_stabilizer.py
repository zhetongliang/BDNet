import numpy as np
import cv2
import torch
import pdb

# Input: list of raw images with size H, W, 4 (RGGB)
# Output: list of stabilized burst
# Note: center frame: middle
def burst_stabilizer_for_raw(burst_list):
    temp_list = [RGGB2Gray(v) for v in burst_list]
    temp_list = [np.power(v, 1.0/2.4) for v in temp_list]
    temp_list = [np.clip(v, 0.0, 1.0) for v in temp_list]
    aligned_burst = burst_list.copy()

    # Pre-define transformation-store array
    n_frames = len(temp_list)
    center_frame_num = int((n_frames - 1)/2)
    transforms = np.zeros((n_frames-1, 3), np.float32)

    # Detect feature points in previous frame
    # pdb.set_trace()
    center_frame = temp_list[center_frame_num]
    center_frame_temp = (center_frame * 255.0).round()
    center_frame_temp = center_frame_temp.astype(np.uint8)
    center_pts = cv2.goodFeaturesToTrack(center_frame_temp, maxCorners=200, qualityLevel=0.01,
                                        minDistance=10, blockSize=3)

    # Calculate optical flow (i.e. track feature points)
    for i in range(n_frames):
        if i == center_frame_num: 
            continue
        curr_frame = temp_list[i]
        # pdb.set_trace()
        center_frame_temp = (center_frame * 255.0).round()
        center_frame_temp = center_frame_temp.astype(np.uint8)
        curr_frame_temp = (curr_frame * 255.0).round()
        curr_frame_temp = curr_frame_temp.astype(np.uint8)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(center_frame_temp, curr_frame_temp, center_pts, None)

        # Sanity check
        assert center_pts.shape == curr_pts.shape
        # Filter only valid points
        idx = np.where(status==1)[0]
        center_pts = center_pts[idx]
        curr_pts = curr_pts[idx]

        #Find transformation matrix
        [m, inliers] = cv2.estimateAffinePartial2D(curr_pts, center_pts)
        curr_frame_stabilized = cv2.warpAffine(aligned_burst[i], m, (curr_frame.shape[1],curr_frame.shape[0]))
        aligned_burst[i] = curr_frame_stabilized
    
    return aligned_burst

# Input: list of raw images with size H, W, 4 (RGGB)
# Output: list of stabilized burst
# Note: center frame: middle
def burst_stabilizer_for_raw_training(burst_list):
    temp_list = [RGGB2Gray(v) for v in burst_list]
    temp_list = [np.power(v, 1.0/2.4) for v in temp_list]
    temp_list = [np.clip(v, 0.0, 1.0) for v in temp_list]
    aligned_burst = burst_list.copy()

    # Pre-define transformation-store array
    n_frames = len(temp_list)
    center_frame_num = int((n_frames - 1)/2)
    transforms = np.zeros((n_frames-1, 3), np.float32)

    # Detect feature points in previous frame
    center_frame = temp_list[center_frame_num]
    center_frame_temp = (center_frame * 255.0).round()
    center_frame_temp = center_frame_temp.astype(np.uint8)
    center_pts = cv2.goodFeaturesToTrack(center_frame_temp, maxCorners=100, qualityLevel=0.01,
                                        minDistance=10, blockSize=3)

    if center_pts is not None:
        # Calculate optical flow (i.e. track feature points)
        for i in range(n_frames):
            if i == center_frame_num: 
                continue
            curr_frame = temp_list[i]
            # pdb.set_trace()
            if len(center_pts) > 10:
                center_frame_temp = (center_frame * 255.0).round()
                center_frame_temp = center_frame_temp.astype(np.uint8)
                curr_frame_temp = (curr_frame * 255.0).round()
                curr_frame_temp = curr_frame_temp.astype(np.uint8)
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(center_frame_temp, curr_frame_temp, center_pts, None)

                idx = np.where(status==1)[0]
                if idx is not None:
                    if len(idx) > 10:
                        center_pts = center_pts[idx]
                        curr_pts = curr_pts[idx]
                        #Find transformation matrix
                        [m, inliers] = cv2.estimateAffinePartial2D(curr_pts, center_pts)
                        temp = aligned_burst[i]
                        temp = temp.astype(np.double)
                        curr_frame_stabilized = cv2.warpAffine(temp, m, (curr_frame.shape[1],curr_frame.shape[0]))
                        curr_frame_stabilized = curr_frame_stabilized.astype(np.float32)
                        # pdb.set_trace()
                        aligned_burst[i] = curr_frame_stabilized
            
    return aligned_burst

# Function: Convert RGGB raw image to Fake Gray image 
def RGGB2Gray(img):
    return np.mean(img, 2)