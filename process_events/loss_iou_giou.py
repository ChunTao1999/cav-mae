# Author: Chun Tao
# built on top of "https://github.com/lilanxiao/Rotated_IoU"

import torch
from torch import Tensor
import numpy as np
from box_intersection_2d import oriented_box_intersection_2d
from min_enclosing_box import smallest_bounding_box
import pdb


def box2corners_th(box:Tensor)-> Tensor:
    """convert box coordinate to corners

    Args:
        box (Tensor): (B, N, 5) with x, y, w, h, alpha
        B - batch size; N - number of boxes in each image

    Returns:
        Tensor: (B, N, 4, 2) corners
    """
    B = box.shape[0]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5] # (B, N, 1)
    x4 = torch.tensor([0.5, -0.5, -0.5, 0.5], dtype=box.dtype).unsqueeze(0).unsqueeze(0).to(box.device) # (1,1,4)
    x4 = x4 * w     # (B, N, 4)
    y4 = torch.tensor([0.5, 0.5, -0.5, -0.5], dtype=box.dtype).unsqueeze(0).unsqueeze(0).to(box.device)
    y4 = y4 * h     # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)     # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)       # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)   # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1,4,2]), rot_T.view([-1,2,2]))
    rotated = rotated.view([B,-1,4,2])          # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated


def cal_iou(box1:Tensor, box2:Tensor):
    """calculate iou

    Args:
        box1 (Tensor): (B, N, 5)
        box2 (Tensor): (B, N, 5)
    
    Returns:
        iou (Tensor): (B, N)
        corners1 (Tensor): (B, N, 4, 2)
        corners1 (Tensor): (B, N, 4, 2)
        U (Tensor): (B, N) area1 + area2 - inter_area
    """
    corners1 = box2corners_th(box1)
    corners2 = box2corners_th(box2)
    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)        #(B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    u = area1 + area2 - inter_area
    iou = inter_area / u
    return iou, corners1, corners2, u


def cal_diou(box1:Tensor, box2:Tensor):
    """calculate diou loss

    Args:
        box1 (Tensor): [description]
        box2 (Tensor): [description]
    """
    iou, corners1, corners2, u = cal_iou(box1, box2)
    w, h = smallest_bounding_box(torch.cat([corners1, corners2], dim=-2))
    c2 = w*w + h*h      # (B, N)
    x_offset = box1[...,0] - box2[..., 0]
    y_offset = box1[...,1] - box2[..., 1]
    d2 = x_offset*x_offset + y_offset*y_offset
    diou_loss = 1. - iou + d2/c2
    return diou_loss, iou


def cal_giou(box1:Tensor, box2:Tensor):
    iou, corners1, corners2, u = cal_iou(box1, box2)
    w, h = smallest_bounding_box(torch.cat([corners1, corners2], dim=-2))
    area_c =  w*h
    giou_loss = 1. - iou + ( area_c - u )/area_c
    return giou_loss, iou


def plot_corners(ax, img, label):
    corners = box2corners_th(label).numpy().squeeze()
    ax.imshow(img, cmap="gray")
    ax.fill(corners[:, 0], corners[:, 1], facecolor="none", edgecolor="r")

