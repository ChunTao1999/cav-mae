import argparse
import bisect
import copy
import cv2
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import PIL
from PIL import Image
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torchvision.transforms as T
from torchvision.utils import save_image
from matplotlib.ticker import PercentFormatter
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import pdb # for debug


def calibrate_camera(objp_w, objp_h,
                     cal_data_path,
                     save_path):
    """calibrate the camera based on finding corners in chessboard photos, and save the camera matrices"""
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points
    objp = np.zeros((objp_w*objp_h,3), np.float32)
    objp[:,:2] = np.mgrid[0:objp_w, 0:objp_h].T.reshape(-1,2)
    # arrays to store object points and image points from all images
    objpoints = []
    imgpoints=[]
    images = glob.glob(os.path.join(cal_data_path, "chessboards", "*.jpg"))
    chessboard_save_path = os.path.join(save_path, "chessboards")
    if not os.path.exists(chessboard_save_path):
        os.makedirs(chessboard_save_path)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find chessboard object point corners
        ret, corners = cv2.findChessboardCorners(gray, (objp_w, objp_h), None) # (img, patternSize)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # draw and display the corners
            cv2.drawChessboardCorners(img, (objp_w,objp_h), corners2, ret)
            cv2.imwrite(os.path.join(chessboard_save_path, fname.split('/')[-1]), img) # optional
    print("\tCorners found in all chessboard images. The rendered images are saved")
    # Compute the 3x3 camera matrix and the distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) # get reversed image shape
    # Save the calibration data
    np.savez(os.path.join(cal_data_path, 'caldata.npz'), x=mtx, y=dist)
    print("\tCamera matrix and distortion coefficients computed and saved")
    # Test undistortion
    img = cv2.imread(images[5])
    h, w = img.shape[:2]
    dst = cv2.undistort(img, mtx, dist)
    cv2.imwrite(os.path.join(chessboard_save_path, images[5].split('/')[-1]+'_undistorted.jpg'), dst)
    print("\tExample chessboard image undistorted and saved")
    return


def define_perspective_transform(cal_data_path, 
                                 img_cal_path,
                                 IMAGE_H,
                                 IMAGE_W,
                                 cv2_imread_frame_dim,
                                 srcpts_arr,
                                 destpts_arr):
    cal_npz = np.load(os.path.join(cal_data_path, 'caldata.npz'))
    mtx, dist = cal_npz['x'], cal_npz['y']
    img_cal = cv2.imread(img_cal_path)
    orig_h, orig_w = img_cal.shape[:2] # 1542, 2314
    img_cal = cv2.undistort(img_cal, mtx, dist)
    cv2.imwrite(os.path.join(cal_data_path, 'cal_img_undistorted.png'), img_cal)
    srcpts = np.float32(srcpts_arr)
    destpts = np.float32(destpts_arr)
    plt.figure()
    plt.imshow(cv2.cvtColor(img_cal, cv2.COLOR_BGR2RGB))
    for (coord_w, coord_h) in srcpts:
        plt.scatter(coord_w, coord_h, color='r', marker='o', s=30, alpha=0.6)
    plt.savefig(os.path.join(cal_data_path, 'cal_img_annotated.png'))
    plt.close()
    # compute PM and IPM matrices and apply to the cal_img
    resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
    resmatrix_inv = cv2.getPerspectiveTransform(destpts, srcpts)
    img_cal_transformed = cv2.warpPerspective(img_cal, resmatrix, (IMAGE_W, IMAGE_H))
    cv2.imwrite(os.path.join(cal_data_path, 'cal_img_bev.png'), img_cal_transformed)
    print("\tPerspective transform defined for BEV, test image results saved")
    # save newly computed PM and IPM matrices
    scale_w, scale_h = cv2_imread_frame_dim[0]/orig_w, cv2_imread_frame_dim[1]/orig_h
    resmatrix = cv2.getPerspectiveTransform(np.float32(srcpts*np.array([scale_w, scale_h])), destpts)
    resmatrix_inv = cv2.getPerspectiveTransform(destpts, np.float32(srcpts*np.array([scale_w, scale_h])))
    np.save(os.path.join(cal_data_path, 'resmatrix.npy'), resmatrix)
    np.save(os.path.join(cal_data_path, 'resmatrix_inv.npy'), resmatrix_inv)
    print("\tPerspective transform and its inverse matrices saved")
    return


def compute_event_loc_dist_curves(event_timeoffset,
                                  event_left,
                                  event_right,
                                  event_len_pix,
                                  frame_image,
                                  frame_dim,
                                  frame_timeoffset,
                                  frame_dist,
                                  wheel_to_base_dist,
                                  base_pixel,
                                  track_width,
                                  veh_speed,
                                  veh_yawrate,
                                  xm_per_pix,
                                  ym_per_pix,
                                  resmatrix_inv,
                                  resmatrix):
    """Rotate and extend the vertical distance, iteratively"""
    start_idx = bisect.bisect_left(veh_speed.iloc[:,0], event_timeoffset)
    end_idx = bisect.bisect_right(veh_speed.iloc[:,0], frame_timeoffset)
    # compute end_idx by distance
    # end_idx = start_idx
    # accum_dist = 0
    # while accum_dist < frame_dist:
    #     accum_dist += veh_speed.iloc[end_idx, 1]*0.01
    #     end_idx += 1                     
    speed_vec, yawrate_vec = veh_speed[start_idx:end_idx].to_numpy(), \
                             veh_yawrate[start_idx:end_idx].to_numpy()
    # accumulate vertical travel distance until it passes wheel to base distance
    # travel_dist = 0
    if event_left==1 and event_right==0:
        current_boxcenter = np.array([frame_dim[0]/2+track_width/2/xm_per_pix, frame_dim[1]-base_pixel+wheel_to_base_dist/ym_per_pix])
    elif event_left==0 and event_right==1:
        current_boxcenter = np.array([frame_dim[0]/2-track_width/2/xm_per_pix, frame_dim[1]-base_pixel+wheel_to_base_dist/ym_per_pix])
    else: # both 1s
        current_boxcenter = np.array([frame_dim[0]/2, frame_dim[1]-base_pixel+wheel_to_base_dist/ym_per_pix])
    vec_idx = 0
    rotate_origin = np.array([frame_dim[0]/2, frame_dim[1]-base_pixel+(wheel_to_base_dist-0.5)/ym_per_pix]) # origin of rotation for the car, in the middle of back wheels
    accum_angle = 0
    accum_dist = 0
    accum_boxcenter = [current_boxcenter]
    while vec_idx < len(speed_vec):
        current_boxcenter[1] -= (speed_vec[vec_idx,1]*0.01 / ym_per_pix) # decrease the distance
        inv_boxcenter = cv2.perspectiveTransform(np.float32([[current_boxcenter]]), resmatrix_inv)[0][0] # (2,)
        inv_rotate_origin = cv2.perspectiveTransform(np.float32([[rotate_origin]]), resmatrix_inv)[0][0]
        # print(inv_boxcenter, inv_rotate_origin)
        inv_current_boxcenter = rotate(origin=inv_rotate_origin,
                                       point=inv_boxcenter,
                                       angle=yawrate_vec[vec_idx,1] * 0.01)
        current_boxcenter = cv2.perspectiveTransform(np.float32([[inv_current_boxcenter]]), resmatrix)[0][0]
        accum_angle += yawrate_vec[vec_idx,1] * 0.01
        accum_boxcenter.append(copy.deepcopy(current_boxcenter))
        vec_idx += 1
    # reshape the bounding box to align to the accumulated angle
    pts_bev = np.float32([[current_boxcenter[0]-event_len_pix/2, current_boxcenter[1]-event_len_pix/2], # topleft
                          [current_boxcenter[0]+event_len_pix/2, current_boxcenter[1]-event_len_pix/2], # topright
                          [current_boxcenter[0]-event_len_pix/2, current_boxcenter[1]+event_len_pix/2], # bottomleft
                          [current_boxcenter[0]+event_len_pix/2, current_boxcenter[1]+event_len_pix/2]]) # bottomright
    pts_bev_rotated = np.zeros_like(pts_bev)
    for idx, point in enumerate(pts_bev):
        rotated_point = rotate(origin=current_boxcenter,
                               point=point,
                               angle=accum_angle,
                               aspect_ratio=1)
        pts_bev_rotated[idx] = rotated_point
    pts_bev_rotated = np.clip(pts_bev_rotated, 10, 590)
    pts_inv = cv2.perspectiveTransform(np.float32([pts_bev_rotated]), resmatrix_inv)[0] # (4,2)
    return pts_bev_rotated, pts_inv, accum_boxcenter


def compute_event_loc_dist_curves_2(event_timeoffset,
                                    event_left,
                                    event_right,
                                    event_len_pix,
                                    frame_image,
                                    frame_dim,
                                    frame_timeoffset,
                                    frame_dist,
                                    wheel_to_base_dist,
                                    base_pixel,
                                    track_width,
                                    veh_speed,
                                    veh_yawrate,
                                    xm_per_pix,
                                    ym_per_pix,
                                    resmatrix_inv,
                                    resmatrix):
    """Second method: map the car's trajectories in BEV"""
    start_idx = bisect.bisect_left(veh_speed.iloc[:,0], event_timeoffset)
    end_idx = bisect.bisect_right(veh_speed.iloc[:,0], frame_timeoffset)
    speed_vec, yawrate_vec = veh_speed[start_idx:end_idx].to_numpy(), \
                             veh_yawrate[start_idx:end_idx].to_numpy()
    if event_left==1 and event_right==0:
        current_boxcenter = np.array([frame_dim[0]/2+track_width/2/xm_per_pix, frame_dim[1]-base_pixel+wheel_to_base_dist/ym_per_pix])
    elif event_left==0 and event_right==1:
        current_boxcenter = np.array([frame_dim[0]/2-track_width/2/xm_per_pix, frame_dim[1]-base_pixel+wheel_to_base_dist/ym_per_pix])
    else: # both 1s
        current_boxcenter = np.array([frame_dim[0]/2, frame_dim[1]-base_pixel+wheel_to_base_dist/ym_per_pix])
    accum_boxcenter = [current_boxcenter]
    # yawrate in radians/sec, positive for left turn, negative for right turn
    headings = np.cumsum(yawrate_vec[:,1]*0.01)
    headings -= headings[-1]
    headings += np.pi / 2 # vertical up is pi/2 radians
    # cum_x_shift = np.sum(np.cos(headings)*speed_vec[:,1]*0.01) # in meters
    # cum_y_shift = -np.sum(np.sin(headings)*speed_vec[:,1]*0.01) # in meters
    # pixel_x_shift = cum_x_shift / xm_per_pix
    # pixel_y_shift = cum_y_shift / ym_per_pix
    # final_boxcenter = current_boxcenter + np.array([pixel_x_shift, pixel_y_shift])
    for vec_idx in np.arange(len(speed_vec))[::-1]:
        current_boxcenter[0] += (np.cos(headings[vec_idx])*speed_vec[:,1][vec_idx]*0.01)/xm_per_pix
        current_boxcenter[1] -= (np.sin(headings[vec_idx])*speed_vec[:,1][vec_idx]*0.01)/ym_per_pix
        accum_boxcenter.append(copy.deepcopy(current_boxcenter))
    pts_bev = np.float32([[current_boxcenter[0]-event_len_pix/2, current_boxcenter[1]-event_len_pix/2], # topleft
                          [current_boxcenter[0]+event_len_pix/2, current_boxcenter[1]-event_len_pix/2], # topright
                          [current_boxcenter[0]-event_len_pix/2, current_boxcenter[1]+event_len_pix/2], # bottomleft
                          [current_boxcenter[0]+event_len_pix/2, current_boxcenter[1]+event_len_pix/2]]) # bottomright
    # pts_bev = np.clip(pts_bev, 10, 590) 
    # pts_inv = cv2.perspectiveTransform(np.float32([pts_bev]), resmatrix_inv)[0]
    # point_rotate = cv2.perspectiveTransform(np.float32([[current_boxcenter]]), resmatrix_inv)[0][0]
    # pts_inv_rotated = np.zeros_like(pts_inv)
    # for idx, point in enumerate(pts_inv):
    #     rotated_point = rotate(origin=point_rotate,
    #                            point=point,
    #                            angle=np.pi/2 - headings[0],
    #                            aspect_ratio=1)
    #     pts_inv_rotated[idx] = rotated_point
    # pts_bev_rotated = cv2.perspectiveTransform(np.float32([pts_inv_rotated]), resmatrix)[0] # (4,2)
    # pts_bev_rotated = np.clip(pts_bev_rotated, 10, 590)
    pts_bev_rotated = np.zeros_like(pts_bev)
    for idx, point in enumerate(pts_bev):
        rotated_point = rotate(origin=current_boxcenter,
                               point=point,
                               angle=(np.pi/2 - headings[0])*(8.8/4.318))
        pts_bev_rotated[idx] = rotated_point
    pts_bev_rotated = np.clip(pts_bev_rotated, 10, 590)
    pts_inv_rotated = cv2.perspectiveTransform(np.float32([pts_bev_rotated]), resmatrix_inv)[0]
    return pts_bev_rotated, pts_inv_rotated, accum_boxcenter


def compute_event_loc_dist_curves_final(event_timeoffset,
                                        event_left,
                                        event_right,
                                        event_len_pix,
                                        event_len_scale,
                                        frame_image,
                                        frame_dim,
                                        frame_timeoffset,
                                        frame_dist,
                                        wheel_to_base_dist,
                                        base_pixel,
                                        track_width,
                                        wheel_width,
                                        wheel_diameter,
                                        veh_speed,
                                        veh_yawrate,
                                        xm_per_pix,
                                        ym_per_pix,
                                        resmatrix_inv,
                                        resmatrix):
    """Final method: 1) trace the center location in BEV 2) compute and apply rotation in world BEV, transform back to BEV and project to RV 3) visualize in BEV and RV"""
    start_idx = bisect.bisect_left(veh_speed.iloc[:,0], event_timeoffset)
    end_idx = bisect.bisect_right(veh_speed.iloc[:,0], frame_timeoffset)
    speed_vec, yawrate_vec = veh_speed[start_idx:end_idx].to_numpy(), \
                             veh_yawrate[start_idx:end_idx].to_numpy()
    if event_left==1 and event_right==0:
        current_boxcenter = np.array([frame_dim[0]/2+track_width/2/xm_per_pix, frame_dim[1]-base_pixel+wheel_to_base_dist/ym_per_pix])
    elif event_left==0 and event_right==1:
        current_boxcenter = np.array([frame_dim[0]/2-track_width/2/xm_per_pix, frame_dim[1]-base_pixel+wheel_to_base_dist/ym_per_pix])
    else: # both 1s
        current_boxcenter = np.array([frame_dim[0]/2, frame_dim[1]-base_pixel+wheel_to_base_dist/ym_per_pix])
    accum_boxcenter = [current_boxcenter]
    # yawrate in radians/sec, positive for left turn, negative for right turn
    headings = np.cumsum(yawrate_vec[:,1]*0.01)
    headings -= headings[-1]
    headings += np.pi / 2 # vertical up is pi/2 radians
    # cum_x_shift = np.sum(np.cos(headings)*speed_vec[:,1]*0.01) # in meters
    # cum_y_shift = -np.sum(np.sin(headings)*speed_vec[:,1]*0.01) # in meters
    # pixel_x_shift = cum_x_shift / xm_per_pix
    # pixel_y_shift = cum_y_shift / ym_per_pix
    # final_boxcenter = current_boxcenter + np.array([pixel_x_shift, pixel_y_shift])
    for vec_idx in np.arange(len(speed_vec))[::-1]:
        current_boxcenter[0] += (np.cos(headings[vec_idx])*speed_vec[:,1][vec_idx]*0.01)/xm_per_pix
        current_boxcenter[1] -= (np.sin(headings[vec_idx])*speed_vec[:,1][vec_idx]*0.01)/ym_per_pix
        accum_boxcenter.append(copy.deepcopy(current_boxcenter))
    pts_bev = np.float32([[current_boxcenter[0]-wheel_width*event_len_scale/2/xm_per_pix, 
                           current_boxcenter[1]-wheel_diameter*event_len_scale/2/ym_per_pix], # topleft
                          [current_boxcenter[0]+wheel_width*event_len_scale/2/xm_per_pix, 
                           current_boxcenter[1]-wheel_diameter*event_len_scale/2/ym_per_pix], # topright
                          [current_boxcenter[0]-wheel_width*event_len_scale/2/xm_per_pix, 
                           current_boxcenter[1]+wheel_diameter*event_len_scale/2/ym_per_pix], # bottomleft
                          [current_boxcenter[0]+wheel_width*event_len_scale/2/xm_per_pix, 
                           current_boxcenter[1]+wheel_diameter*event_len_scale/2/ym_per_pix]]) # bottomright
    pt_bev_center = np.float32(current_boxcenter)
    pt_world_bev_center = pt_bev_center * np.array([xm_per_pix, ym_per_pix])
    pts_bev_rot = np.zeros_like(pts_bev)
    for idx, point in enumerate(pts_bev):
        # 1. First convert the BEV coordinates to world BEV coordinates in meters
        # 2. Apply the rotation matrix with computed $\angle \alpha_{total}$ in the BEV and find shift for each vertice in meters
        # 3. Convert the shift in meters to shift in pixels, so that the same rotation can be applied in our BEV frame
        # 4. Once the bounding box in the BEV frame is rotated, project to the RV frame using the IPM matrix
        # pdb.set_trace()
        point_world_bev = point * np.array([xm_per_pix, ym_per_pix])
        rot_point_world_bev = rotate(origin=pt_world_bev_center,
                                     point=point_world_bev,
                                     angle=(np.pi/2 - headings[0])) # the rotated point has units of meters
        rot_point_bev = rot_point_world_bev / np.array([xm_per_pix, ym_per_pix])                        
        pts_bev_rot[idx] = rot_point_bev
    pts_bev_rot = np.clip(pts_bev_rot, 5, 595) # rotated rect box in bev
    pts_inv_rot = cv2.perspectiveTransform(np.float32([pts_bev_rot]), resmatrix_inv)[0] # poly in rv
    return pts_bev_rot, pts_inv_rot, accum_boxcenter


def rotate(origin, point, angle): # angle in radians
    """Rotate a point cclw by a given angle around a given origin point"""
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def add_bbox_to_frame(image, pts):
    image_height, image_width, image_ch = image.shape
    debug_image = copy.deepcopy(image)
    pts = pts[[0,1,3,2], :] # in order to draw in the correct order across points
    pts = pts.reshape((-1, 1, 2))
    debug_image = cv2.polylines(img=debug_image,
                                pts=np.int32([pts]),
                                isClosed=True,
                                color=[255,255,255],
                                thickness=10)
    return debug_image


def add_accum_boxcenter(image, accum_boxcenter): # a list
    """Plot the trajectory of each boxcenter in the BEV"""
    for boxcenter in accum_boxcenter:
        image = cv2.circle(image,
                           tuple(np.int_(boxcenter)),
                           radius=5,
                           color=(0,0,255),
                           thickness=2)
    return image


def bbox_coords_to_bbox_label(pts_inv):
    """Convert from 4 bbox coordinates to a bbox label representation (topleft_x, topleft_y, width, height), needs to be json-serializable too"""
    bbox_label = pts_inv.tolist()
    return bbox_label


def normalize_wheel_accel(wheelAccel_seg, v_peak, v_nom):
    wheelAccel_seg /= np.square(v_peak/v_nom)
    return wheelAccel_seg


# 9.10
def settling_time_and_dist(wheelAccel_seg, veh_speed, event_timestamp, timespan, SSV, threshold): # SSV stands for "Steady-State Value"
    """Compute settling time and distance of the wheelAccel segment, based on a percentage of peak to peak amplitude"""
    wheelAccel_seg = wheelAccel_seg[0]
    if wheelAccel_seg.shape[0] == 0:
        return 0, 0, 0
    amp_pp = np.max(wheelAccel_seg) - np.min(wheelAccel_seg)# peak to peak amplitude
    tol_threshold = threshold * amp_pp
    # use SSV +- tol_threshold as the check condition for the signal amplitude at any instant
    midpoint = wheelAccel_seg.shape[0] // 2
    start = midpoint
    for idx, sig_val in enumerate(wheelAccel_seg[midpoint:]):
        if sig_val > (SSV+tol_threshold) or sig_val < (SSV-tol_threshold):
            start = idx + midpoint
    settling_time = (start-midpoint) * 0.002
    # use extracted veh_speed segment to compute settling_dist
    speed_vec = veh_speed.iloc[bisect.bisect_left(veh_speed.iloc[:,0], event_timestamp):bisect.bisect_right(veh_speed.iloc[:,0], event_timestamp+settling_time), 1]
    settling_dist = 0.01 * np.sum(speed_vec)
    peaks_pos, _ = find_peaks(wheelAccel_seg[midpoint:], prominence=1)
    peaks_neg, _ = find_peaks(-wheelAccel_seg[midpoint:], prominence=1)
    try:
        p2p_time = np.abs(peaks_pos[0]-peaks_neg[0]) * 0.002
    except:
        p2p_time = 0
    return settling_time, settling_dist, p2p_time


def plot_image(batch_imgs, gt_bboxes, gt_labels, pred_bboxes, pred_labels, label_dict, save_path):
    """Used during training and inference to visualize bbox regression and label classification progress"""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(50, 10))
    plt.subplots_adjust(wspace=0, hspace=0)
    batch_imgs = batch_imgs.detach().cpu()
    gt_bboxes = gt_bboxes.detach().cpu()
    gt_labels = gt_labels.detach().cpu()
    pred_bboxes = pred_bboxes.detach().cpu()
    pred_labels = pred_labels.detach().cpu()

    for j in range(batch_imgs.shape[0]):
        image_arr = batch_imgs[j].permute(1,2,0).numpy()
        debug_img = (image_arr * 255).astype(np.uint8).copy()
        gt_bbox = np.array(gt_bboxes[j], np.int32).reshape((-1, 1, 2))
        gt_bbox = gt_bbox[:, :, [1,0]] # reverse width and height
        gt_bbox = gt_bbox[[0,1,3,2], :, :] # order of traversal these points
        gt_label = str(gt_labels.tolist()[j])
        pred_label = str(pred_labels.tolist()[j])

        pred_bbox = np.array(pred_bboxes[j], np.int32).reshape((-1, 1, 2))
        pred_bbox = pred_bbox[:, :, [1,0]]
        pred_bbox = pred_bbox[[0,1,3,2], :, :]

        img_annotated = cv2.polylines(img=debug_img,
                                      pts=[gt_bbox],
                                      isClosed=True,
                                      color=[255,255,255],
                                      thickness=5)
        img_annotated = cv2.polylines(img=img_annotated,
                                      pts=[pred_bbox],
                                      isClosed=True,
                                      color=[255,255,0],
                                      thickness=5)
        img_annotated = cv2.putText(img_annotated,
                                    label_dict[gt_label],
                                    (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, 
                                    (255,255,255),
                                    3)
        img_annotated = cv2.putText(img_annotated,
                                    label_dict[pred_label],
                                    (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, 
                                    (255,255,0),
                                    3)

        axes[j//2][j%2].imshow(img_annotated)
    
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return


def plot_conf_matrix(exp_path, true_labels_list, pred_labels_list):
    if len(true_labels_list) == 0 or len(pred_labels_list) == 0:
        print("Empty classification label lists, confusion matrix skipped......")
    else:
        true_labels = torch.cat(true_labels_list, dim=0)
        predicted_labels = torch.cat(pred_labels_list, dim=0)
        conf_matrix = confusion_matrix(true_labels.numpy(), predicted_labels.numpy(),)
        class_names = ['Pothole', 'Manhole Cover', 'Drain Gate', 'Unknown', 'Speed Bump']  # Replace with your actual class names
        class_dict = {i: class_name for i, class_name in enumerate(class_names)}
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=[class_dict[i] for i in range(len(class_names))],
                    yticklabels=[class_dict[i] for i in range(len(class_names))])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(exp_path, "conf_mat.png"))
        plt.close()
    return


def save_loss_tallies(exp_path, train_loss_tally, test_loss_tally, train_acc_tally, test_acc_tally):
    np.save(os.path.join(exp_path, "train_loss.npy"), np.array(train_loss_tally))
    np.save(os.path.join(exp_path, "test_loss.npy"), np.array(test_loss_tally))
    np.save(os.path.join(exp_path, "train_acc.npy"), np.array(train_acc_tally))
    np.save(os.path.join(exp_path, "test_acc.npy"), np.array(test_acc_tally))
    return


def plot_loss_curves(exp_path):
    train_loss_tally = np.load(os.path.join(exp_path, "train_loss.npy"))
    test_loss_tally = np.load(os.path.join(exp_path, "test_loss.npy"))
    train_acc_tally = np.load(os.path.join(exp_path, "train_acc.npy"))
    test_acc_tally = np.load(os.path.join(exp_path, "test_acc.npy"))
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(len(train_loss_tally)), train_loss_tally, 'b-', linewidth=2.0, label='Train loss')
    ax1.plot(np.arange(len(test_loss_tally))*len(train_loss_tally)//len(test_loss_tally), test_loss_tally, 'r-', linewidth=2.0, label='Test loss')
    ax1.set_ylabel('Loss', color=(139/255, 69/255, 19/255), fontsize=12) # brown
    ax1.tick_params('y', color=(139/255, 69/255, 19/255))
    ax1.set_xlabel('Iteration', color=(139/255, 69/255, 19/255), fontsize=12)
    ax1.tick_params('x', color=(139/255, 69/255, 19/255))
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(train_acc_tally))*len(train_loss_tally)//len(test_loss_tally), train_acc_tally, 'b-', linewidth=2.0, label='Train acc')
    ax2.plot(np.arange(len(test_acc_tally))*len(train_loss_tally)//len(test_loss_tally), test_acc_tally, 'r-', linewidth=2.0, label='Test acc')
    ax2.set_ylabel('Accuracy', color=(139/255, 69/255, 19/255), fontsize=12)
    ax2.tick_params('y', color=(139/255, 69/255, 19/255))
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper right')
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.title('Loss and Accuracy', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, "loss_acc.png"))
    plt.close()
    return


def polygon_area(vertices):
    """"
    Calculate the polygon area using the Shoelace formula
    Args: vertices (Tensor): shape (num_vertices, 2), ordered either in cloc
    Returns: area (Tensor): area of the polygon
    """
    x, y = vertices[:,0], vertices[:,1]
    area = 0.5 * torch.abs(torch.sum(x[:-1] * y[1:]) + x[-1] * y[0] - torch.sum(y[:-1] * x[1:]) - y[-1] * x[0])
    return area


def compute_intersection_area(pred_vertices, target_vertices):
    """
    Calculate intersection area between pairs of predicted and target polygon indices
    Args: pred_vertices (Tensor): predicted poly indices, shape (batch_size, num_vertices=4, 2)
          target_vertices (Tensor): targeted poly indices, shape (batch_size, num_vertices=4, 2)
    Returns: intersection_areas (Tensor): shape (batch_size, )
    """
    bs = pred_vertices.shape[0]
    intersection_areas = torch.zeros(bs)
    pred_vertices = pred_vertices.view(bs, -1, 2) # (bs, 4, 2)
    target_vertices = target_vertices.view(bs, -1, 2)
    # for each data sample
    for i in range(bs):
        # try:
        area1 = polygon_area(pred_vertices[i])
        area2 = polygon_area(target_vertices[i])
        intersection_vertices = torch.cat([pred_vertices[i], target_vertices[i]])
        # intersection_vertices = torch.tensor(intersection_vertices[ConvexHull(intersection_vertices.clone().detach().numpy()).vertices])
        intersection_area = polygon_area(intersection_vertices)
        intersection_area = torch.min(intersection_area, torch.min(area1, area2))
        intersection_areas[i] = intersection_area
    return intersection_areas


def compute_union_area(pred_vertices, target_vertices, intersection_areas):
    """
    Calculate union area
    """
    total_areas_predicted = torch.stack([polygon_area(poly.view(-1, 2)) for poly in pred_vertices])
    total_areas_target = torch.stack([polygon_area(poly.view(-1, 2)) for poly in target_vertices])
    union_areas = total_areas_predicted + total_areas_target - intersection_areas
    return union_areas


def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)


def split_train_val_test(metafile_path, data_folder_path, seed, split_ratio):
    # use train_test_split 
    with open(metafile_path, 'r') as f:
        data_dict = json.load(f)
    frame_names = []
    for event_id, value in data_dict.items():
        for fn in value['frames']:
            frame_names.append(fn)
    train_files, test_files = train_test_split(frame_names, test_size=split_ratio[1]+split_ratio[2], random_state=seed)
    val_files, test_files = train_test_split(test_files, test_size=split_ratio[2]/(split_ratio[1]+split_ratio[2]), random_state=seed)
    
    split_dict = {}
    for frame_name in frame_names:
        if frame_name in train_files:
            split_dict[frame_name] = 0
        elif frame_name in val_files:
            split_dict[frame_name] = 1
        else:
            split_dict[frame_name] = 2
    return split_dict


def calculate_normalization(metafile_path, data_folder_path, split_dict):
    # iterate over images described in the metafile, in the data folder. make a tensor for all images, calculate mean and std over the entire dataset
    frame_transform = T.Compose([T.Resize(size=[256,256], 
                                          interpolation=T.InterpolationMode.BILINEAR),
                                 T.ToTensor()])
    count_frames = 0
    with open(metafile_path, 'r') as f:
        data_dict = json.load(f)
    for event_id, value in data_dict.items():
        for fn in value['frames']:
            if split_dict[fn] == 0:
                event_frame = Image.open(os.path.join(data_folder_path, 'undistorted_rv', fn+'.png'))
                event_frame_tensor = frame_transform(event_frame)
                if count_frames == 0:
                    all_frame_tensor = event_frame_tensor.unsqueeze(0)
                else:
                    all_frame_tensor = torch.cat((all_frame_tensor, event_frame_tensor.unsqueeze(0)), dim=0)
                count_frames += 1
                if count_frames % 100 == 0:
                    print(f"Progress: {count_frames}")
            else:
                continue
    print(f"Training set size: {count_frames:d}")
    all_frame_tensor = all_frame_tensor.permute(1, 0, 2, 3).contiguous().view(3, -1)
    return all_frame_tensor.mean(dim=1), all_frame_tensor.std(dim=1)


def visualize_frame_bbox(frame_tensor, mean, std, bbox_tensor, save_name):
    unnormalize = T.Normalize(mean=[-m/s for m,s in zip(mean, std)], std=[1/s for s in std])
    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(unnormalize(frame_tensor).permute(1, 2, 0).detach().cpu().numpy()) # (h, w, c)
    if bbox_tensor.shape[0] > 0:
        bbox = _corners(*bbox_tensor)
        ax.fill(bbox[:, 0], bbox[:, 1], facecolor="none", edgecolor="r")
    ax.axis("off")
    plt.savefig(fname=os.path.join("visualize", save_name+".png"))
    return 


def _rotate(points: np.ndarray, theta: float) -> np.ndarray:
    """Rotates the points counterclockwise by multiplying by the rotation matrix, around the origin"""
    return points @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _corners(pos_x: float, pos_y: float, yaw: float, width: float, height: float) -> np.ndarray:
    points = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]).astype(np.float)
    points *= np.array([width, height]) / 2
    points = _rotate(points, yaw)
    points += np.array([pos_x, pos_y])
    return points
