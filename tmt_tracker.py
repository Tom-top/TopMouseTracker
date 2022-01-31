#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:03:44 2018

@author: tomtop
"""

import os
import time

import cv2
import numpy as np
import skvideo.io

import tmt_utilities as utils
import tmt_io as IO


class TopMouseTracker:
    """Main tracking class :

       Methods :

           • __init__ : Sets all the parameters usefull for the tracking,
           displaying and saving processes.

           • SetRegistrationParameters : Computes and sets variables for depth
           registration based on the size of RGB and DEPTH images.

           • SetMetaDataParameters : Loads all the metadata from excel files :
           The 'Mice_Video_Info.xlsx' file that contains all timing data for
           each animal. The 'MetaData*.xlsx' file that contains all video meta-
           data.

           • SetROI : Displays the first frame from the video for the user to
           select the ROI where the segmentation has to run. Press 'r' to reset
           the ROI; Press 'c' to set the ROI.

           • Main : Retrieves the next RGB and DEPTH frames and runs the
           segmentation on them.

           • RunSegmentations : Wrapper to run the Mouse segmentation, the DEPTH
           registration, and the Cotton segmentation in one method call

           • RunSegmentationMouse : Segments the Mouse position. Steps :

               1) RGB --> HSV (Hue, Saturation, Value) LUT conversion
               2) Gaussian blur
               3) Binarization based on HSV parameters
               4) Opening to filter salt and pepper noise
               5) Closing to fill the objects with holes
               6) Contour detection on the binary mask
               7) Detection of the biggest contour
               8) Computation of the centroid of the largest detected object

           • RunSegmentationCotton : Segments the Cotton average height. Steps :

               * Using the blurred HSV image from RunSegmentationMouse Method
               1) Binarization based on HSV parameters
               2) Opening to filter salt and pepper noise
               3) Closing to fill the objects with holes
               4) Contour detection on the binary mask
               5) Bitwise AND operation between binary mask and registered depth
               6) Computing the average height values of the pixels inside the Bitwise AND mask
               7) Computing the number of Large/Small objects based on contour area sizes

           • RegisterDepth : Registers DEPTH image onto the RGB image

           • CreateDisplay : Creates the canvas for the segmentation display
           and saving

           • WriteDisplay : Writes metadata information on the created canvas

           • SaveTracking : Saves tracking as a video

           • ComputeDistanceTraveled : Computes the distance travelled between
           the previous and the current position in (cm)

           • UpdatePosition : Updates the position of the mouse

           • StorePosition : Stores the position of the mouse

           • Error : Stores the number of times that the program failed to detect
           the Mouse

           • ReturnTracking : Returns the tracking canvas for display purposes

    Parameters :

        **kwargs (dict) : dictionnary with all the parameters useful for the tracker
    """

    def __init__(self, color_capture, depth_capture, **kwargs):

        # General variables
        # ----------------------------------------------------------------------
        self._args = kwargs  # Loads the main arguments
        self._tracker_break = False  # Checks if the tracker was manually aborted to resume segmentation
        self._tracker_stop = False  # Trigger to stop segmentation when video is empty
        self._time_segmentation_start = None  # Time at which the segmentation starts
        self._time_segmentation_end = None  # Time at which the segmentation ends

        self._animal_tag = self._args["general"]["general"]["animal_tag"]  # Loads the name of the mouse
        self._cage_width = self._args["general"]["cage"]["width"]
        self._cage_length = self._args["general"]["cage"]["length"]
        self._color_capture = color_capture
        self._test_color_frame = self._color_capture.get_frame(
            self._args["tracking"]["general"]["test_frame"])  # Loads a test color frame
        self._height_color, self._width_color = self._test_color_frame.shape[0], \
                                                self._test_color_frame.shape[1]  # Height, Width of the color frames

        if self._args["tracking"]["general"]["segment_nesting"]:
            self._depth_capture = depth_capture
            self._test_depth_frame = self._depth_capture.get_frame(
                self._args["tracking"]["general"]["test_frame"])  # Loads a test DEPTH frame
            self._height_depth, self._width_depth = self._test_depth_frame.shape[0], \
                                                    self._test_depth_frame.shape[1]

        self._framerate_color = color_capture.fps
        self._duration_between_two_frames = 1 / self._framerate_color

        # Global tracking variables
        # ----------------------------------------------------------------------
        self._positions = []  # Position of the mouse on every frame
        self._nm_average_height = []  # Height of the nesting material on every frame
        self._errors = 0  # Counter for the number of times that the trackers fails to segment the animal

        # MetaData variables
        # ----------------------------------------------------------------------
        self._recording_start = self._args["general"]["video"]["start"]
        self._recording_end = self._args["general"]["video"]["end"]
        if self._recording_end <= self._recording_start:
            raise RuntimeError("End video has to be superior to Start: see config.cfg")
        self._recording_duration = self._recording_end - self._recording_start

        # Real-Time tracking variables
        # ----------------------------------------------------------------------
        self.frame_number = self._recording_start * self._framerate_color
        self.fn = 0
        self.current_animal_position = []  # Position of the mouse on every frame
        self.center = None  # Centroid (x,y) of the animal on every frame
        self.corrected_center = None  # Corrected centroid (x,y) of the animal on every frame

        # Tracking canvas variables
        # ----------------------------------------------------------------------
        self.contour_thickness = 2  # thickness of the contours
        self.center_size = 3  # size of the object centroid
        self.center_thickness = 5  # thickness of the object centroid

        # Saving variables
        # ----------------------------------------------------------------------
        self._start_saving = False

    def set_segmentation_saving(self):
        if self._args["tracking"]["display"]["save_tracking_display"]:
            self.video_string = "tracking_{0}.{1}".format(self._animal_tag,
                                                          self._args["tracking"]["display"]["saving_extension"])
            self.test_canvas = np.zeros((self._width_color_cropped, self._height_color_cropped))
            self.test_canvas = cv2.resize(self.test_canvas, (0, 0),
                                          fx=1. / self._args["tracking"]["display"]["resize_tracking_window"],
                                          fy=1. / self._args["tracking"]["display"]["resize_tracking_window"])

            self.video_writer = skvideo.io.FFmpegWriter(
                os.path.join(self._args["general"]["segmentation_directory"], self.video_string),
                inputdict={
                    '-r': str(self._framerate_color),
                },
                outputdict={
                    '-r': str(self._framerate_color),
                })

    def set_segmentation_roi(self, roi_ref_points=None, pos_frame=0):
        """
        Method that displays a test frame from the video for the user to select the ROI in which
        the segmentation will be run
        :param roi_ref_points:
        :return:
        """

        utils.print_color_message("[INFO] Press R to reset ROI, and C to crop the selected ROI", "bold")
        test_frame = self._color_capture.get_frame(pos_frame)  # Loads a test RGB frame
        test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)

        if roi_ref_points is None:
            self._roi_ref_points = IO.CroppingROI(test_frame).get_roi()  # Defining the ROI for segmentation
        else:
            self._roi_ref_points = roi_ref_points

        self.up_left_x = int(self._roi_ref_points[0][0])  # Defines the Up Left ROI corner X coordinates
        self.up_left_y = int(self._roi_ref_points[0][1])  # Defines the Up Left ROI corner Y coordinates
        self.low_right_x = int(self._roi_ref_points[1][0])  # Defines the Low Right ROI corner X coordinates
        self.low_right_y = int(self._roi_ref_points[1][1])  # Defines the Low Right ROI corner Y coordinates

        self.roi_width = abs(self.low_right_x - self.up_left_x)  # Computes the ROI width in (px)
        self.roi_length = abs(self.low_right_y - self.up_left_y)  # Computes the ROI length in (px)

        self.distance_ratio = (abs(self.up_left_x - self.low_right_x) / self._args["general"]["cage"]["length"] +
                               abs(self.up_left_y - self.low_right_y) / self._args["general"]["cage"]["width"]) / 2

        self._test_frame_color_cropped = self._test_color_frame[self.up_left_y:self.low_right_y,
                                         self.up_left_x:self.low_right_x]  # Creates a croped test frame according to ROI
        self._height_color_cropped, self._width_color_cropped = self._test_frame_color_cropped.shape[0], \
                                                                self._test_frame_color_cropped.shape[
                                                                    1]  # Computes the height/width of the cropped frame

    def nothing(self, x):
        """
        Empty method
        :param x:
        :return:
        """
        pass

    def adjust_thresholding(self, pos_frame, which='animal'):
        """Method to check the segmentation parameters for the animal/material

        Args : capture (cap) : capture of the video to be analyzed
               framePos (int) : number of the frame to be analyzed
               which (str) : 'animal' if the mouse segmentation parameters have to be tested
                             'object' if the cotton segmentation parameters have to be tested

        The trackbars represent respectively :

            The Hue values Low/High
            The Saturation values Low/High
            The Value values Low/High

        If the elements in the image become black : those elements will become 0 after binarization, i.e : neglected
        If the elements in the image remain colored : those elements will become 1 after binarization, i.e : segmented
        """

        cv2.namedWindow('Adjust Thresholding')
        if which == 'animal':
            cv2.createTrackbar('H_Low', 'Adjust Thresholding',
                               self._args["tracking"]["thresholding"]["min_animal"][0],
                               255,
                               self.nothing)
            cv2.createTrackbar('H_High', 'Adjust Thresholding',
                               self._args["tracking"]["thresholding"]["max_animal"][0],
                               255,
                               self.nothing)
            cv2.createTrackbar('S_Low', 'Adjust Thresholding',
                               self._args["tracking"]["thresholding"]["min_animal"][1],
                               255,
                               self.nothing)
            cv2.createTrackbar('S_High', 'Adjust Thresholding',
                               self._args["tracking"]["thresholding"]["max_animal"][1],
                               255,
                               self.nothing)
            cv2.createTrackbar('V_Low', 'Adjust Thresholding',
                               self._args["tracking"]["thresholding"]["min_animal"][2],
                               255,
                               self.nothing)
            cv2.createTrackbar('V_High', 'Adjust Thresholding',
                               self._args["tracking"]["thresholding"]["max_animal"][2],
                               255,
                               self.nothing)
        elif which == 'material':
            cv2.createTrackbar('H_Low', 'Adjust Thresholding',
                               self._args["tracking"]["thresholding"]["min_material"][0],
                               255,
                               self.nothing)
            cv2.createTrackbar('H_High', 'Adjust Thresholding',
                               self._args["tracking"]["thresholding"]["max_material"][0],
                               255,
                               self.nothing)
            cv2.createTrackbar('S_Low', 'Adjust Thresholding',
                               self._args["tracking"]["thresholding"]["min_material"][1],
                               255,
                               self.nothing)
            cv2.createTrackbar('S_High', 'Adjust Thresholding',
                               self._args["tracking"]["thresholding"]["max_material"][1],
                               255,
                               self.nothing)
            cv2.createTrackbar('V_Low', 'Adjust Thresholding',
                               self._args["tracking"]["thresholding"]["min_material"][2],
                               255,
                               self.nothing)
            cv2.createTrackbar('V_High', 'Adjust Thresholding',
                               self._args["tracking"]["thresholding"]["max_material"][2],
                               255,
                               self.nothing)
        else:
            utils.print_color_message("[INFO] Select 'animal' or 'material' to preview the default thresholding values",
                                      "darkgreen")
            cv2.createTrackbar('H_Low', 'Adjust Thresholding', 0, 255, self.nothing)
            cv2.createTrackbar('H_High', 'Adjust Thresholding', 255, 255, self.nothing)
            cv2.createTrackbar('S_Low', 'Adjust Thresholding', 0, 255, self.nothing)
            cv2.createTrackbar('S_High', 'Adjust Thresholding', 255, 255, self.nothing)
            cv2.createTrackbar('V_Low', 'Adjust Thresholding', 0, 255, self.nothing)
            cv2.createTrackbar('V_High', 'Adjust Thresholding', 255, 255, self.nothing)

        test_frame = self._color_capture.get_frame(pos_frame)
        test_frame_cropped = test_frame[self.up_left_y:self.low_right_y, self.up_left_x:self.low_right_x]
        test_frame_cropped_hsv = cv2.cvtColor(test_frame_cropped, cv2.COLOR_BGR2HSV)
        test_frame_blurred = cv2.blur(test_frame_cropped_hsv, (5, 5))

        while True:
            h_l = cv2.getTrackbarPos('H_Low', 'Adjust Thresholding')
            h_h = cv2.getTrackbarPos('H_High', 'Adjust Thresholding')
            s_l = cv2.getTrackbarPos('S_Low', 'Adjust Thresholding')
            s_h = cv2.getTrackbarPos('S_High', 'Adjust Thresholding')
            v_l = cv2.getTrackbarPos('V_Low', 'Adjust Thresholding')
            v_h = cv2.getTrackbarPos('V_High', 'Adjust Thresholding')
            test_mask_mouse = cv2.inRange(test_frame_blurred, (h_l, s_l, v_l), (h_h, s_h, v_h))
            overlay = cv2.bitwise_and(test_frame_cropped_hsv, test_frame_cropped_hsv, mask=test_mask_mouse)
            cv2.imshow('Adjust Thresholding', overlay)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
        for i in range(1, 5):
            cv2.waitKey(1)

    def main(self):
        try:
            self.color_frame = self._color_capture.get_frame(
                self.frame_number / self._framerate_color)  # Reads the following frame from the video capture
            if self._args["tracking"]["general"]["segment_nesting"]:  # If the cotton segmentation mode was selected
                self.depth_frame = self._depth_capture.get_frame(
                    self.frame_number / self._framerate_color)  # Reads the following frame from the video capture
            if not self._tracker_stop:
                self.current_time = int(self.frame_number / self._framerate_color)  # Sets the time
                if self._recording_start * self._framerate_color <= self.frame_number <= self._recording_end * self._framerate_color:
                    self.run_segmentations()
            self.fn += 1
            self.frame_number += 1
        except IndexError:
            print("All frames: ({0}), from the video were analyzed".format(self.fn))
            self.frame_number = 0
            self._tracker_stop = True

    def run_segmentations(self):
        self.run_segmentation_animal()
        if self._args["tracking"]["general"]["segment_nesting"]:
            self.run_segmentation_nesting_material()
        if self._args["tracking"]["display"]["show_tracking"] or self._args["tracking"]["display"][
            "save_tracking_display"]:
            self.create_segmentation_display()
        if self._args["tracking"]["display"]["save_tracking_display"]:
            if not self._start_saving:
                self._start_saving = True

    def run_segmentation_animal(self):
        self.clone_color_frame = self.color_frame.copy()
        self.clone_color_frame = cv2.cvtColor(self.clone_color_frame,
                                              cv2.COLOR_BGR2RGB)
        self.cropped_color_frame = self.color_frame[self.up_left_y:self.low_right_y,
                                   self.up_left_x:self.low_right_x]
        self.color_display = cv2.cvtColor(self.cropped_color_frame,
                                          cv2.COLOR_BGR2RGB)
        self.hsv_frame = cv2.cvtColor(self.cropped_color_frame,
                                      cv2.COLOR_BGR2HSV)
        self.blur = cv2.blur(self.hsv_frame, (5, 5))
        self.mask_animal = cv2.inRange(self.blur,
                                       np.array(self._args["tracking"]["thresholding"]["min_animal"]),
                                       np.array(self._args["tracking"]["thresholding"]["max_animal"]))
        self.opening_animal = cv2.morphologyEx(self.mask_animal, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8),
                                               iterations=1)
        self.closing_animal = cv2.morphologyEx(self.opening_animal, cv2.MORPH_CLOSE,
                                               np.ones((5, 5), np.uint8),
                                               iterations=1)
        self.contours_animal = cv2.findContours(self.closing_animal.copy(),
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        if self.contours_animal != []:
            self.largest_contour = max(self.contours_animal, key=cv2.contourArea)
            self.area = cv2.contourArea(self.largest_contour)
            if self.area > self._args["tracking"]["filtering"]["animal_minimal_size"] \
                    and self.area < self._args["tracking"]["filtering"]["animal_maximal_size"]:
                ((self.x, self.y), self.radius) = cv2.minEnclosingCircle(self.largest_contour)
                self.moments = cv2.moments(self.largest_contour)
                self.center = (int(self.moments["m10"] / self.moments["m00"]),
                               int(self.moments["m01"] / self.moments["m00"]))
                self.remember_previous_position()
            else:
                self.error()
                self.remember_previous_position()
        else:
            self.error()
            self.area = 0
            self.remember_previous_position()

    def run_segmentation_nesting_material(self):
        self.mask_nesting_material = cv2.inRange(self.blur,
                                                 np.array(self._args["tracking"]["thresholding"]["min_material"]),
                                                 np.array(self._args["tracking"]["thresholding"][
                                                              "max_material"]))  # Thresholds the image to binary
        self.opening_nesting_material = cv2.morphologyEx(self.mask_nesting_material,
                                                         cv2.MORPH_OPEN,
                                                         np.ones((5, 5), np.uint8),
                                                         iterations=1)
        self.closing_nesting_material = cv2.morphologyEx(self.opening_nesting_material,
                                                         cv2.MORPH_CLOSE,
                                                         np.ones((5, 5), np.uint8),
                                                         iterations=1)
        self.contours_nesting_material = cv2.findContours(self.closing_nesting_material.copy(),
                                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        self.contours_nesting_material_filtered = [cnt for cnt in self.contours_nesting_material if
                                                   cv2.contourArea(cnt) >=
                                                   self._args["tracking"]["filtering"]["nesting_material_size_filter"]]
        self.final_mask_nesting_material = np.array(
            cv2.drawContours(np.zeros(self.closing_nesting_material.shape),
                             self.contours_nesting_material_filtered,
                             -1,
                             (255, 255, 255),
                             cv2.FILLED), dtype="uint8")
        self.cropped_depth_frame = self.depth_frame[self.up_left_y:self.low_right_y, self.up_left_x:self.low_right_x]
        self.bitwise_depth_nesting_material = cv2.bitwise_and(self.final_mask_nesting_material,
                                                              self.cropped_depth_frame[:, :, 0])
        self.average_pixel_intensity_nesting_material = self.bitwise_depth_nesting_material[
            np.nonzero(self.bitwise_depth_nesting_material)]
        try:
            self.average_pixel_intensity_nesting_material = int(np.mean(self.average_pixel_intensity_nesting_material))
        except:
            self.average_pixel_intensity_nesting_material = 0
        self._nm_average_height.append(self.average_pixel_intensity_nesting_material)

    def correct_first_positions(self):
        self._positions = np.array(self._positions)
        positions_to_correct = np.where(self._positions == None)[0]
        if positions_to_correct:
            idx_first_correct_position = np.max(positions_to_correct) + 1
            self._positions[0:idx_first_correct_position-1] = self._positions[idx_first_correct_position]

    def create_segmentation_display(self):
        self.write_display()
        self.clone_color_frame[self.up_left_y:self.up_left_y + self.color_display.shape[0],
        self.up_left_x:self.up_left_x + self.color_display.shape[1]] = self.color_display
        cv2.circle(self.clone_color_frame, self.corrected_center, self.center_size, (0, 0, 255),
                   self.center_thickness)
        # cv2.rectangle(self.clone_color_frame, (self.up_left_x, self.up_left_y), (self.low_right_x, self.low_right_y),
        #               (255, 0, 0),
        #               self.contour_thickness)
        self.h_stack = self.clone_color_frame[self.up_left_y:self.low_right_y,
                       self.up_left_x:self.low_right_x]
        self.h_stack = cv2.resize(self.h_stack, (0, 0),
                                  fx=1. / self._args["tracking"]["display"]["resize_tracking_window"],
                                  fy=1. / self._args["tracking"]["display"]["resize_tracking_window"])

    def write_display(self):
        if self.area > self._args["tracking"]["filtering"]["animal_minimal_size"] \
                and self.area < self._args["tracking"]["filtering"]["animal_maximal_size"]:
            cv2.drawContours(self.color_display, [self.largest_contour], 0, (0, 0, 255),
                             self.contour_thickness)
        if self._args["tracking"]["general"]["segment_nesting"]:
            cv2.drawContours(self.color_display,
                             self.contours_nesting_material_filtered,
                             -1,
                             (0, 255, 0),
                             self.contour_thickness)

    def write_tracking(self):
        if self._start_saving:
            self.h_stack = cv2.cvtColor(self.h_stack, cv2.COLOR_BGR2RGB)
            self.video_writer.writeFrame(self.h_stack)

    def update_animal_position(self):
        if len(self.current_animal_position) == 0:
            self.current_animal_position.append(self.center)
        elif len(self.current_animal_position) == 1:
            self.current_animal_position.append(self.center)
        elif len(self.current_animal_position) == 2:
            self.current_animal_position[0] = self.current_animal_position[1]
            self.current_animal_position[1] = self.center

    def remember_previous_position(self):
        self._positions.append(self.center)
        if self.center is not None:
            self.update_animal_position()
            self.corrected_center = (self.center[0] + self.up_left_x,
                                     self.center[1] + self.up_left_y)

    def error(self):
        self._errors += 1
        if self._args["tracking"]["display"]["show_tracking"]:
            if self._errors == 1:
                print("[WARNING] No contour detected, assuming old position !")
            elif self._errors % 100 == 0:
                print("[WARNING] No contour detected, assuming old position !")

    def return_tracking(self):
        if self._args["tracking"]["display"]["show_tracking"]:
            try:
                self.new_h_stack = cv2.cvtColor(self.h_stack, cv2.COLOR_BGR2RGB)
                self.new_h_stack = cv2.resize(self.new_h_stack, (0, 0),
                                              fx=self._args["tracking"]["display"]["resize_tracking_window"],
                                              fy=self._args["tracking"]["display"]["resize_tracking_window"])
                return self.new_h_stack
            except:
                return np.array([])
        else:
            return np.array([])

    def segment(self):
        if not self._tracker_break:
            utils.print_color_message("[INFO] Starting segmentation for mouse {0}".
                                      format(self._args["general"]["general"]["animal_tag"]), "darkgreen")
        else:
            utils.print_color_message("[INFO] Resuming segmentation for mouse {0}".
                                      format(self._args["general"]["general"]["animal_tag"]), "magenta")
        self._tracker_break = False
        self._time_segmentation_start = time.time()
        start_time = time.localtime()
        h, m, s = start_time.tm_hour, start_time.tm_min, start_time.tm_sec
        self.set_segmentation_saving()
        utils.print_color_message("[INFO] Segmentation started at : {0}h {1}m {2}s".format(h, m, s), "darkgreen")
        n_frames = (self._recording_end * self._framerate_color) - (
                self._recording_start * self._framerate_color)
        if self._args["tracking"]["general"]["segment_nesting"]:
            try:
                while True:
                    self.main()
                    if not self._tracker_stop:
                        if self._args["tracking"]["display"]["save_tracking_display"]:
                            self.write_tracking()
                        if self._args["tracking"]["display"]["show_tracking"]:
                            segmentation = self.return_tracking()
                            if segmentation.size != 0:
                                cv2.imshow('segmentation', segmentation)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                        if not self._recording_end == 0:
                            if self.fn % (int(0.01 * n_frames)) == 0:
                                utils.print_color_message(
                                    'Loaded and analyzed : ' + str(self.fn) + '/' + str(int(n_frames)) + \
                                    ' = ' + str(round((float(self.fn) / float((n_frames))) * 100)) \
                                    + '% frames', "darkgreen")
                                utils.print_color_message(
                                    utils.print_loading_bar(round((float(self.fn) / float(n_frames)) * 100)),
                                    "darkgreen")
                            if self.frame_number == int(self._recording_end * self._framerate_color):
                                self.frame_number = 0
                                break
                    else:
                        self.frame_number = 0
                        break
            except KeyboardInterrupt:
                self._tracker_break = True
        else:
            try:
                while True:
                    self.main()
                    if not self._tracker_stop:
                        if self._args["tracking"]["display"]["save_tracking_display"]:
                            self.write_tracking()
                        if self._args["tracking"]["display"]["show_tracking"]:
                            segmentation = self.return_tracking()
                            if segmentation:
                                cv2.imshow('segmentation', segmentation)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    self._tracker_break = True
                                    break
                        if not self._recording_end == 0:
                            if self.fn % (int(0.01 * n_frames)) == 0:
                                utils.print_color_message(
                                    'Loaded and analyzed : ' + str(self.fn) + '/' + str(int(n_frames)) + \
                                    ' = ' + str(round((float(self.fn) / float((n_frames))) * 100)) \
                                    + '% frames', "darkgreen")
                                utils.print_color_message(
                                    utils.print_loading_bar(round((float(self.fn) / float(n_frames)) * 100)),
                                    "darkgreen")
                            if self.frame_number == int(self._recording_end * self._framerate_color):
                                self.frame_number = 0
                                break
                    else:
                        self.frame_number = 0
                        break
            except KeyboardInterrupt:
                self._tracker_break = True
        if not self._tracker_break:
            utils.print_color_message("[INFO] Animal {0} has been successfully analyzed"
                                      .format(self._animal_tag), "darkgreen")
        else:
            utils.print_color_message("[INFO] Tracking for Animal {0} has been aborted"
                                      .format(self._animal_tag), "darkred")
        if self._args["tracking"]["display"]["show_tracking"]:
            cv2.destroyAllWindows()
        if self._args["tracking"]["display"]["save_tracking_display"]:
            self.video_writer.close()
        self._time_segmentation_end = time.time()
        diff = self._time_segmentation_end - self._time_segmentation_start
        h, m, s = utils.get_h_m_s(diff)
        self.correct_first_positions()
        utils.print_color_message("[INFO] Segmentation started at : {0}h {1}m {2}s".format(h, m, s), "darkgreen")
        self.save_tracking()

    def save_tracking(self):
        np.save(os.path.join(self._args["general"]["segmentation_directory"],
                             "data_{0}_ref_points.npy".format(self._animal_tag)),
                self._roi_ref_points)
        np.save(os.path.join(self._args["general"]["segmentation_directory"],
                             "data_{0}_positions.npy".format(self._animal_tag)),
                self._positions)

        if self._args["tracking"]["general"]["segment_nesting"]:
            np.save(os.path.join(self._args["general"]["segmentation_directory"],
                                 "data_{0}_nesting_material_average_height.npy".format(self._animal_tag)),
                    self._nm_average_height)