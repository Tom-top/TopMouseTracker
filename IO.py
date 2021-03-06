#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:00:30 2019

@author: tomtop
"""

import os;
import subprocess;
import shlex;
import numpy as np;
import cv2;
from natsort import natsorted;
import skvideo.io;
import moviepy.editor as mpy;
from moviepy.editor import VideoFileClip, concatenate_videoclips

import TopMouseTracker.Parameters as params;
import TopMouseTracker.Settings as settings;
import TopMouseTracker.Utilities as utils;

def concatenate_video_clips(folder_path) :

    video_files = natsorted([os.path.join(folder_path, i) for i in os.listdir(folder_path) if os.path.splitext(i)[1] == ".mpg" and os.path.splitext(i)[0][0] != "."])
    print(np.array(video_files))
    video_clips = [VideoFileClip(i) for i in video_files]
    video_clip = concatenate_videoclips(video_clips)
    print("\n")
    print(video_clip.duration)
    
    test_frame = video_clip.get_frame(0)
    
    return test_frame, [video_clip], video_clips

def VideoConverter(directory,**kwargs) :
    
    '''Function that converts all videos in the directory into the right format
    for analysis
    
    Params :
        directory (str) : The directory where the videos to be converted are
        kwargs (dict) : dictionnary holding parameters for the VideoConverter function
        (see videoParameters in Kinect.py file for more info)
    '''
        
    Videosources = [];
    Videosinks = [];
    counter = 0;
        
    for files in natsorted(os.listdir(directory)) :
        
        if files.split('.')[-1] == 'mpg' :
            
            Videosources.append(directory+str(files));
            Videosinks.append(directory+'Mpeg_'+str(counter)+'.mp4');
            counter+=1;
            
        else :
            
            utils.PrintColoredMessage("[WARNING] File {0} is not in the right format".format(files), "darkred");
    
    for videosource,videosinks in zip(Videosources,Videosinks) :
        
        name = videosource.split('/')[-1];
        args = shlex.split(kwargs['runstr'].format(kwargs['handBrakeCLI'],\
                           videosource,videosinks,kwargs['encoder'],\
                           kwargs['quality'],kwargs['framerate']));
        subprocess.call(args,shell=False);
        
        print('\n');
        utils.PrintColoredMessage('###############################################',"darkgreen");
        utils.PrintColoredMessage("[INFO] Video {0} has been successfully converted".format(name),"darkgreen");
        utils.PrintColoredMessage('###############################################',"darkgreen");
              
        if kwargs['playSound'] :
            utils.PlaySound(2,params.sounds['Purr']);


def GetColorFrame(stream,RF):
    
    '''Function that takes the stream of RGB images from the kinect as an argument
    and returns the image as an array
    
    Params :
        stream (kinect_stream) : stream from kinect
        RF (int/float) : resizing factor 
        
    Returns :
        _frameRGB (array) : RGB image of each frame coming from the stream
    '''
    
    _frameRGB = stream.get_last_color_frame();
    
    _W,_H = stream.color_frame_desc.Width,stream.color_frame_desc.Height;
    _frameRGB = _frameRGB.reshape(_H,_W,-1).astype(np.uint8);
    _frameRGB = cv2.resize(_frameRGB, (int(_W/RF),int(_H/RF)));
    
    return _frameRGB;


def GetDepthFrame(stream,RF):
    
    '''Function that takes the stream of DEPTH images from the kinect as an argument
    and returns the image as an array
    
    Params :
        stream (kinect_stream) : stream from kinect
        RF (int/float) : resizing factor 
        
    Returns :
        _frameDEPTH (array) : DEPTH image of each frame coming from the stream
    '''
    
    _frameDEPTH = stream.get_last_depth_frame()
    
    W,H = stream.depth_frame_desc.Width,stream.depth_frame_desc.Height
    _frameDEPTH = _frameDEPTH.reshape((H, W,-1)).astype(np.uint16)
    _frameDEPTH = cv2.resize(_frameDEPTH, (int(W/RF),int(H/RF)))
    
    return _frameDEPTH;


def VideoLoader(directory, in_folder=False, **kwargs) :
    
    '''Function that loads all the video from a directory and returns 
    a testFram for ROI selection, and the captures
    
    Params :
        directory (str) : directory where the videos to load are
        kwargs (dict) : dictionnary holding parameters for the videoLoader function
        (see videoParameters in Kinect.py file for more info)
        
    Returns :
        Captures (capture) : captures from all videos in directory
        testFrame (array) : test frame that is loaded for the croppingROI function
    '''
    
    RGBTrigger = False;
    DEPTHTrigger = False;
    
    RGBCaptures = [];
    DEPTHCaptures = [];
    RGBTestFrame = None;
    DEPTHTestFrame = None;
    
    print("\n");
    
    if not in_folder :
    
        for folder in natsorted(os.listdir(directory)) :
            
            dirPath = os.path.join(directory,folder);
            
            if os.path.isdir(dirPath) and dirPath in kwargs["main"]["workingDir"] :
        
                for file in natsorted(os.listdir(os.path.join(directory,folder))) :
                    
                    if file.split('.')[-1] == kwargs["main"]["extensionLoad"] :
                        
                        if file.split("_")[0] == kwargs["main"]["rgbVideoName"] :
    #                    if file.split('_')[0] == "Raw" :
                            #cap = cv2.VideoCapture(os.path.join(directory,file));
                            cap = mpy.VideoFileClip(os.path.join(dirPath,file));
                            #cap = skvideo.io.vreader(os.path.join(directory,file));
                            RGBCaptures.append(cap);
                            
                            if not RGBTrigger :
                            
                                frame = cap.get_frame(kwargs["main"]["testFramePos"]);
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);
                                
                                RGBTestFrame = frame;
                                RGBTrigger = True;
            
                            utils.PrintColoredMessage("[INFO] {0} loaded successfully".format(file),"darkgreen");
                                  
                            if kwargs["main"]["playSound"] :
                                  
                                try :  
                                    utils.PlaySound(1,params.sounds['Purr']);
                                except :
                                    pass;
                            
                        elif file.split("_")[0] == kwargs["main"]["depthVideoName"] :
                            
                            #cap = cv2.VideoCapture(os.path.join(directory,file));
                            cap = mpy.VideoFileClip(os.path.join(dirPath,file));
                            #cap = skvideo.io.vreader(os.path.join(directory,file));
                            DEPTHCaptures.append(cap);
        
                            if not DEPTHTrigger :
                                
                                frame = cap.get_frame(kwargs["main"]["testFramePos"]);
                                
                                DEPTHTestFrame = frame;
                                DEPTHTrigger = True;
                            
                            utils.PrintColoredMessage("[INFO] {0} loaded successfully".format(file),"darkgreen");
                                  
                            if kwargs["main"]["playSound"] :
                            
                                try :  
                                    utils.PlaySound(1,params.sounds['Purr']);
                                except :
                                    pass;
                                    
    elif in_folder :
        
        for file in natsorted(os.listdir(directory)) :
            
            if file.split('.')[-1] == kwargs["main"]["extensionLoad"] :
                
                if file.split("_")[0] == kwargs["main"]["rgbVideoName"] :

                    cap = mpy.VideoFileClip(os.path.join(directory,file))
                    RGBCaptures.append(cap)
                    
                    if not RGBTrigger :
                    
                        frame = cap.get_frame(kwargs["main"]["testFramePos"]);
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);
                        
                        RGBTestFrame = frame
                        RGBTrigger = True
    
                    utils.PrintColoredMessage("[INFO] {0} loaded successfully".format(file),"darkgreen");
                          
                    if kwargs["main"]["playSound"] :
                          
                        try :  
                            
                            utils.PlaySound(1,params.sounds['Purr'])
                            
                        except :
                            
                            pass
                
                elif file.split("_")[0] == kwargs["main"]["depthVideoName"] :
                    
                    cap = mpy.VideoFileClip(os.path.join(directory,file))
                    DEPTHCaptures.append(cap)

                    if not DEPTHTrigger :
                        
                        frame = cap.get_frame(kwargs["main"]["testFramePos"])
                        
                        DEPTHTestFrame = frame
                        DEPTHTrigger = True
                    
                    utils.PrintColoredMessage("[INFO] {0} loaded successfully".format(file),"darkgreen")
                          
                    if kwargs["main"]["playSound"] :
                    
                        try :  
                            
                            utils.PlaySound(1,params.sounds['Purr'])
                            
                        except :
                            
                            pass
    
    if not RGBTrigger and not DEPTHTrigger:
        
        utils.PrintColoredMessage("[WARNING] Sorry, no video file in the right format was found","darkred")
            
    return RGBCaptures,DEPTHCaptures,RGBTestFrame,DEPTHTestFrame


class CroppingROI():
    
    '''Class that takes a an image as an argument and allows the user to draw a
    ROI for the segmentation.
    
    Params :
        frame (np.array) : image where the ROI has to be drawn
        
    Returns :
        refPt : the combination of coordinates ((x1,y1),(x2,y2)) of the Top Left
        corner and Low Right corner of the ROI, respectively
    '''
    
    def __init__(self, frame):
        
        utils.PrintColoredMessage("\n[INFO] Select the ROI for segmentation","darkgreen");
        
        self.refPt = [];
        self.frame = frame.copy();
        self.W, self.H, _ = self.frame.shape
        cv2.namedWindow("image", cv2.WINDOW_NORMAL);
        cv2.resizeWindow("image", self.H,self.W);
        cv2.setMouseCallback("image", self.clickAndCrop);
        self.clone = self.frame.copy();
        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.moveWindow("image", 0, 0);
            cv2.imshow("image", self.frame);
            key = cv2.waitKey(10) & 0xFF;
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                utils.PrintColoredMessage("[INFO] ROI was reset","darkgreen");
                self.refPt = [];
                self.frame = self.clone.copy();
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                utils.PrintColoredMessage("[INFO] ROI successfully set","darkgreen");
                break;
        # close all open windows
        cv2.destroyAllWindows();
        for i in range (1,5):
            cv2.waitKey(1);
    
    def clickAndCrop(self, event, x, y, flags, param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)];
        elif event == cv2.EVENT_LBUTTONUP:
            self.refPt.append((x, y));
        if len(self.refPt) == 2 and self.refPt != None :
            cv2.rectangle(self.frame, self.refPt[0], self.refPt[1], (0, 0, 255), 2);
            cv2.imshow("image", self.frame);
            
    def roi(self) :
        
        return self.refPt;