#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:00:30 2019

@author: Thomas TOPILKO
"""

import os
import cv2
from natsort import natsorted
from moviepy.editor import VideoFileClip, concatenate_videoclips

import tmt_utilities as utils

def concatenate_video_clips(name_prefix, path=None, **config):
    if path is None:
        video_files = natsorted([os.path.join(config["general"]["recording_directory"], i)
                                 for i in os.listdir(config["general"]["recording_directory"]) if
                                 os.path.splitext(i)[1] == ".{}".format(config["general"]["general"]["extension_video"])
                                 and os.path.splitext(i)[0][0] != "." and os.path.splitext(i)[0].startswith(name_prefix)])
    else:
        video_files = natsorted([os.path.join(path, i)
                                 for i in os.listdir(path) if
                                 os.path.splitext(i)[1] == ".{}".format(config["general"]["general"]["extension_video"])
                                 and os.path.splitext(i)[0][0] != "." and os.path.splitext(i)[0].startswith(name_prefix)])
    video_clips = [VideoFileClip(i) for i in video_files]
    video_clip = concatenate_videoclips(video_clips)
    return video_clip

class CroppingROI:
    """Class that takes a an image as an argument and allows the user to draw a
    ROI for the segmentation.

    Params :
        frame (np.array) : image where the ROI has to be drawn

    Returns :
        refPt : the combination of coordinates ((x1,y1),(x2,y2)) of the Top Left
        corner and Low Right corner of the ROI, respectively
    """
    def __init__(self, frame):
        utils.print_color_message("\n[INFO] Select the ROI for segmentation", "darkgreen")
        self.roi_ref_points = []
        self.frame = frame.copy()
        self.frame_clone = self.frame.copy()
        self.W, self.H, _ = self.frame.shape
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", self.H, self.W)
        cv2.setMouseCallback("image", self.click_and_crop)
        while True:
            cv2.moveWindow("image", 0, 0)
            cv2.imshow("image", self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                utils.print_color_message("[INFO] ROI was reset", "darkgreen")
                self.roi_ref_points = []
                self.frame = self.frame_clone.copy()
            elif key == ord("c"):
                utils.print_color_message("[INFO] ROI successfully set", "darkgreen")
                break
        cv2.destroyAllWindows()
        for i in range(1, 5):
            cv2.waitKey(1)

    def click_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_ref_points = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_ref_points.append((x, y))
        if len(self.roi_ref_points) == 2 and self.roi_ref_points is not None:
            cv2.rectangle(self.frame, self.roi_ref_points[0], self.roi_ref_points[1], (0, 0, 255), 2)
            cv2.imshow("image", self.frame)

    def get_roi(self):
        return self.roi_ref_points