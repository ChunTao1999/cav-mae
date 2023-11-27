# ==============================================================================================
# Title: find_min_rect_bbox.py
# Author: Chun Tao
# Date: 09-25-2023
# Description: Transforms arbitrarily-shaped polygons into tight rectangle bounding boxes
# Based on: https://github.com/dbworth/minimum-area-bounding-rectangle/blob/master/python/min_bounding_rect.py
# ==============================================================================================

import argparse
import cv2
import json
import math
import numpy as np
from numpy import *
import os
import sys
# from utils import find_rotated_bbox_bev
import pdb # for debug


def get_rotating_caliper_bbox_list(hull_points_2d):
    """
    Args:
    hull_points_2d: array of hull points. each element should have [x,y] format
    Returns:
    bbox_list: list of rotated rectangles for the given convex hull points
    """
    # Compute edges (x2-x1,y2-y1)
    edges = np.zeros( (len(hull_points_2d)-1,2) ) # empty 2 column array
    for i in range( len(edges) ):
        edge_x = hull_points_2d[i+1,0] - hull_points_2d[i,0]
        edge_y = hull_points_2d[i+1,1] - hull_points_2d[i,1]
        edges[i] = [edge_x,edge_y]
    # Calculate edge angles   atan2(y/x)
    edge_angles = np.zeros( (len(edges)) ) # empty 1 column array
    for i in range( len(edge_angles) ):
        edge_angles[i] = np.arctan2( edges[i,1], edges[i,0] )
    # Check for angles in 1st quadrant
    for i in range( len(edge_angles) ):
        edge_angles[i] = np.abs( edge_angles[i] % (np.pi/2) ) # want strictly positive answers
    #print "Edge angles in 1st Quadrant: \n", edge_angles
    # Remove duplicate angles
    edge_angles = np.unique(edge_angles)
    #print "Unique edge angles: \n", edge_angles
    bbox_list=[]
    for i in range( len(edge_angles) ):
        # Create rotation matrix to shift points to baseline
        # R = [ cos(theta)      , cos(theta-PI/2)
        #       cos(theta+PI/2) , cos(theta)     ]
        R = np.array([ [ np.cos(edge_angles[i]), np.cos(edge_angles[i]-(np.pi/2)) ], [ np.cos(edge_angles[i]+(np.pi/2)), np.cos(edge_angles[i]) ] ])
        # Apply this rotation to convex hull points
        rot_points = np.dot(R, np.transpose(hull_points_2d) ) # 2x2 * 2xn
        # Find min/max x,y points
        min_x = np.nanmin(rot_points[0], axis=0)
        max_x = np.nanmax(rot_points[0], axis=0)
        min_y = np.nanmin(rot_points[1], axis=0)
        max_y = np.nanmax(rot_points[1], axis=0)
        # Calculate height/width/area of this bounding rectangle
        width = max_x - min_x
        height = max_y - min_y
        area = width*height
        # Calculate center point and restore to original coordinate system
        center_x = (min_x + max_x)/2
        center_y = (min_y + max_y)/2
        center_point = np.dot( [ center_x, center_y ], R )
        # Calculate corner points and restore to original coordinate system
        corner_points = np.zeros( (4,2) ) # empty 2 column array
        corner_points[0] = np.dot( [ max_x, min_y ], R )
        corner_points[1] = np.dot( [ min_x, min_y ], R )
        corner_points[2] = np.dot( [ min_x, max_y ], R )
        corner_points[3] = np.dot( [ max_x, max_y ], R )
        bbox_info = [edge_angles[i], area, width, height, min_x, max_x, min_y, max_y, corner_points, center_point]
        bbox_list.append(bbox_info)
    return bbox_list


def minBoundingRect(hull_points_2d):
    #print "Input convex hull points: "
    #print hull_points_2d

    # Compute edges (x2-x1,y2-y1)
    edges = zeros( (len(hull_points_2d)-1,2) ) # empty 2 column array
    for i in range( len(edges) ):
        edge_x = hull_points_2d[i+1,0] - hull_points_2d[i,0]
        edge_y = hull_points_2d[i+1,1] - hull_points_2d[i,1]
        edges[i] = [edge_x,edge_y]
    #print "Edges: \n", edges

    # Calculate edge angles   atan2(y/x)
    edge_angles = zeros( (len(edges)) ) # empty 1 column array
    for i in range( len(edge_angles) ):
        edge_angles[i] = math.atan2( edges[i,1], edges[i,0] )
    #print "Edge angles: \n", edge_angles

    # Check for angles in 1st quadrant
    for i in range( len(edge_angles) ):
        edge_angles[i] = abs( edge_angles[i] % (math.pi/2) ) # want strictly positive answers
    #print "Edge angles in 1st Quadrant: \n", edge_angles

    # Remove duplicate angles
    edge_angles = unique(edge_angles)
    #print "Unique edge angles: \n", edge_angles

    # Test each angle to find bounding box with smallest area
    min_bbox = (0, sys.maxsize, 0, 0, 0, 0, 0, 0) # rot_angle, area, width, height, min_x, max_x, min_y, max_y
    print("Testing", len(edge_angles), "possible rotations for bounding box... \n")
    for i in range( len(edge_angles) ):

        # Create rotation matrix to shift points to baseline
        # R = [ cos(theta)      , cos(theta-PI/2)
        #       cos(theta+PI/2) , cos(theta)     ]
        R = array([ [ math.cos(edge_angles[i]), math.cos(edge_angles[i]-(math.pi/2)) ], [ math.cos(edge_angles[i]+(math.pi/2)), math.cos(edge_angles[i]) ] ])
        #print "Rotation matrix for ", edge_angles[i], " is \n", R

        # Apply this rotation to convex hull points
        rot_points = dot(R, transpose(hull_points_2d) ) # 2x2 * 2xn
        #print "Rotated hull points are \n", rot_points

        # Find min/max x,y points
        min_x = nanmin(rot_points[0], axis=0)
        max_x = nanmax(rot_points[0], axis=0)
        min_y = nanmin(rot_points[1], axis=0)
        max_y = nanmax(rot_points[1], axis=0)
        #print "Min x:", min_x, " Max x: ", max_x, "   Min y:", min_y, " Max y: ", max_y

        # Calculate height/width/area of this bounding rectangle
        width = max_x - min_x
        height = max_y - min_y
        area = width*height
        #print "Potential bounding box ", i, ":  width: ", width, " height: ", height, "  area: ", area 

        # Store the smallest rect found first (a simple convex hull might have 2 answers with same area)
        if (area < min_bbox[1]):
            min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )
        # Bypass, return the last found rect
        #min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )

    # Re-create rotation matrix for smallest rect
    angle = min_bbox[0]   
    R = array([ [ math.cos(angle), math.cos(angle-(math.pi/2)) ], [ math.cos(angle+(math.pi/2)), math.cos(angle) ] ])
    #print "Projection matrix: \n", R

    # Project convex hull points onto rotated frame
    proj_points = dot(R, transpose(hull_points_2d) ) # 2x2 * 2xn
    #print "Project hull points are \n", proj_points

    # min/max x,y points are against baseline
    min_x = min_bbox[4]
    max_x = min_bbox[5]
    min_y = min_bbox[6]
    max_y = min_bbox[7]
    #print "Min x:", min_x, " Max x: ", max_x, "   Min y:", min_y, " Max y: ", max_y

    # Calculate center point and project onto rotated frame
    center_x = (min_x + max_x)/2
    center_y = (min_y + max_y)/2
    center_point = dot( [ center_x, center_y ], R )
    #print "Bounding box center point: \n", center_point

    # Calculate corner points and project onto rotated frame
    corner_points = zeros( (4,2) ) # empty 2 column array
    corner_points[0] = dot( [ max_x, min_y ], R )
    corner_points[1] = dot( [ min_x, min_y ], R )
    corner_points[2] = dot( [ min_x, max_y ], R )
    corner_points[3] = dot( [ max_x, max_y ], R )
    #print "Bounding box corner points: \n", corner_points

    #print "Angle of rotation: ", angle, "rad  ", angle * (180/math.pi), "deg"

    return (angle, min_bbox[1], min_bbox[2], min_bbox[3], center_point, corner_points) # rot_angle, area, width, height, center_point, corner_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', type=str, default='', required=True, help='path to the data folder')
    parser.add_argument('-c', '--cal-data-path', type=str, default='', required=True, help='path containing calibration and perspective transform parameters')
    parser.add_argument('-p', '--prev-json-path', type=str, default='', required=True, help='path to previously saved meta json file')
    parser.add_argument('-w', '--track-width', type=float, default=0.5, required=True, help='width of the bbox in world coordinate, in meters')
    parser.add_argument('--xm-per-pix', type=float, default=4.318/500, required=False, help='x meters per pixel in the BEV')
    parser.add_argument('--ym-per-pix', type=float, default=8.8/330, required=False, help='y meters per pixel in the BEV')
    parser.add_argument('--bev-frame-size', type=lambda s: tuple(map(int, s.split(','))), default=(600,600), required=True, help='defined BEV frame size; tuple of integers')
    args = parser.parse_args()

    # Import the metafile as dict
    with open(args.prev_json_path, 'r') as in_file:
        event_dict = json.load(in_file)
    event_id_list = list(event_dict.keys())

    # Import perspective transform
    resmatrix, resmatrix_inv = np.load(os.path.join(args.cal_data_path, 'resmatrix.npy')), \
                               np.load(os.path.join(args.cal_data_path, 'resmatrix_inv.npy'))

    # For each event_id
    for event_id in event_id_list:
        frame_names = event_dict[event_id]["frames"]
        # For each frame
        for frame_idx, frame_name in enumerate(frame_names):
            frame_path = os.path.join(args.data_path, "undistorted_rv", frame_name+".png")
            frame_rv = cv2.imread(frame_path) # (1080, 1920, 3)
            # Perspective transform the frame
            frame_bev = cv2.warpPerspective(np.float32(frame_rv),
                                            resmatrix,
                                            args.bev_frame_size)
            
            frame_bev_annotated = cv2.imread(os.path.join(args.data_path, "undistorted_bev_annotated", frame_name+".png"))
            cv2.imwrite(f"frame_bev_annotated_{frame_idx}.png", frame_bev_annotated)
           

            polygon_box_arr = np.array(event_dict[event_id]['polygon_box'][frame_idx])
            min_bbox = minBoundingRect(polygon_box_arr) # rot_angle, area, width, height, center_point, corner_points
            frame_rv_annotated = cv2.imread(os.path.join(args.data_path, "undistorted_rv_annotated", frame_name+".png"))
            cv2.rectangle(img=frame_rv_annotated,
                          pt1=np.int_(min_bbox[-1][1]),
                          pt2=np.int_(min_bbox[-1][-1]),
                          color=(0,0,255),
                          thickness=3)
            cv2.imwrite(f"frame_rv_annotated_minbox_{frame_idx}.png", frame_rv_annotated)


            # Compute bbox vertice's coords based on boxcenter in BEV and the pre-defined square size, assuming square shape in real-world coordinate

            pdb.set_trace()
            # 

        pdb.set_trace()
    
    pdb.set_trace()