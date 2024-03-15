import copy
import glob
import cv2 as cv
import VisualOdometry as vo
import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()
gt = vo.loadGroundTruth()
left = vo.loadImages('left')
right = vo.loadImages('right')
P0 = vo.loadProjectionMatrix('left')
K,R_w_0,t_w_0 = vo.decomposeProjectionMatrix(P0)


C0 = vo.initialState(R_w_0,t_w_0)
Cn = C0

scale = 1.0
baseline = 0.54
prev_frame_left = left[0]
prev_frame_right = right[0]

poses = []
translation = [Cn[0][3],Cn[1][3],Cn[2][3]]
poses.append(translation)

estimation = '3d2d'

for i in range(1,len(left)):

    if(estimation=='2d2d'):

        curr_frame_left = left[i]

        kp0, des0 = vo.feature_detection_extraction(prev_frame_left)
        kp1, des1 = vo.feature_detection_extraction(curr_frame_left)

        matches = vo.feature_matching(des0, des1)
        good = vo.filter_matches_distance(matches, 0.75)

        scale = vo.absoluteScale(gt[i - 1][0], gt[i - 1][1], gt[i - 1][2], gt[i][0], gt[i][1], gt[i][2])

        Rk, tk, _, _ = vo.estimate_motion(good, kp0, kp1, K)
        tk = np.multiply(tk, scale)

        Tk = vo.transformationMatrix(Rk, tk)
        Tk = np.linalg.inv(Tk) # findEssentialMat daje obratnu transformaciju - pa treba ovo Tk invertovati
        Cn = np.matmul(Cn, Tk)  # novo Cn

        translation = [Cn[0][3], Cn[1][3], Cn[2][3]]
        poses.append(translation)

        print(i)

        # ----------------------------------------------------------
        prev_frame_left = curr_frame_left


    elif(estimation=='3d2d'):

        curr_frame_left = left[i]
        curr_frame_right = right[i]

        # Depth se racuna za prethodni frame (vidi estimate_motion matricu sta prima u opisu)
        disparity = vo.disparityMap(prev_frame_left,prev_frame_right)
        depth = vo.depthMap(disparity,K,baseline)

        kp0, des0 = vo.feature_detection_extraction(prev_frame_left)
        kp1, des1 = vo.feature_detection_extraction(curr_frame_left)

        matches = vo.feature_matching(des0, des1)
        good = vo.filter_matches_distance(matches, 0.75)

        Rk, tk, _, _ = vo.estimate_motion(good,kp0,kp1,K,depth)

        Tk = vo.transformationMatrix(Rk, tk)
        Tk = np.linalg.inv(Tk)  # findEssentialMat daje obratnu transformaciju - pa treba ovo Tk invertovati
        Cn = np.matmul(Cn, Tk)  # novo Cn

        translation = [Cn[0][3], Cn[1][3], Cn[2][3]]
        poses.append(translation)

        print(i)

        # ----------------------------------------------------------
        prev_frame_left = curr_frame_left
        prev_frame_right = curr_frame_right







x_gt,y_gt,z_gt = vo.coordinates(gt)
x_vo,y_vo,z_vo = vo.coordinates(poses)
vo.plot_both(x_gt,z_gt,x_vo,z_vo,'KITTI_GT_VO_3D2D_0')

stop = time.time()
print(stop-start)



