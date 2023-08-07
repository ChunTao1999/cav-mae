import argparse
import cv2
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess
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
    # save the computed PM and IPM matrices
    np.save(os.path.join(cal_data_path, 'resmatrix.npy'), resmatrix)
    np.save(os.path.join(cal_data_path, 'resmatrix_inv.npy'), resmatrix_inv)
    print("\tPerspective transform and its inverse matrices saved")
    return