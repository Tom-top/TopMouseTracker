#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:03:44 2018

@author: tomtop
"""

import os;
import cv2;
import numpy as np;
import pandas as pd;
from math import sqrt;

import TopMouseTracker.Parameters as params;
import TopMouseTracker.Utilities as utils;
import TopMouseTracker.IO as IO;


class TopMouseTracker():
    
    '''Class that holds all parameters usefull for the segmentation
    
    Params :
        frame (array) : testFrame coming from the VideoLoader function
        data (list) : List holding all the metadata for each mouse coming from
        the excel spreadsheet
    '''
    
    def __init__(self,**kwargs) :
        
        #General variables
        #----------------------------------------------------------------------
        self._args = kwargs; #Arguments
        self._mouse = self._args["mouse"]; #Name of the mouse
        
        videoInfoWorkbook = pd.read_excel(self._args["baseDir"]+'/Mice_Video_Info.xlsx'); #Load video info excel sheet
        videoInfo = videoInfoWorkbook.as_matrix(); #Transforms it into a matrix
        
        for metaData in videoInfo :
            if str(metaData[0]) == self._args["mouse"] :
                self._tStart = int(metaData[1]); #Moment at which the cotton is added (s)
                self._tStartBehav = int(metaData[2]); #Moment at which the mouse start nest-building (s)
                self._tEnd = [int(metaData[3]),int(metaData[4]),\
                             int(metaData[5]),int(metaData[6])]; #Length of each video of the experiment (max 4 videos)
        
        self._nVideos = 0; #Variable holding the number of videos to be analyzed
        
        for i in self._tEnd :
            if i != 0 :
                self._nVideos+=1;
        
        #Global tracking variables
        #----------------------------------------------------------------------
        self._positions = []; #Position of the mouse every s/frameRate
        self._maskAreas = []; #Size of the mouse mask every s/frameRate
        self._distance = 0.; #Cumulative distance traveled by the mouse
        self._errors = 0; #Counter for the number of times that the trackers fails to segment the animal
        
        #Real-Time tracking variables
        #----------------------------------------------------------------------
        self.frameNumber = 0;
        self.videoNumber = 0;
        self.realTimePosition = []; #Position of the mouse in real time
        self.realTimeSpeed = 0.; #Speed of the mouse in real time
        self.center = None; #Centroid (x,y) of the mouse binary mask in real time
        self.correctedCenter = None; #Corrected centroid (x,y) of the mouse binary mask in real time
        self.trackingStream = [];
        
    def SetROI(self) :
        
        self._refPt = IO.CroppingROI(self._args["testFrame"][0]).roi(); #Defining the ROI for segmentation
        
    def OffLineTracker(self):
        
        #Get frame from capture
        #----------------------------------------------------------------------
        self.ret, self.RGBFrame = self._args["captures"][self.videoNumber].read(); #Reads the following frame from the video capture
        self.frameNumber += 1; #Increments the frame number variable
        self.time = self.frameNumber/self._args["framerate"]; #Sets the time
           
        #If capture still has frames, and the following frame was successfully retrieved
        #----------------------------------------------------------------------
        if self.ret : #If the next frame was well retrieved
            
            if self.videoNumber == 0 : #If the first video is being processed
                
                if self.time >= self._tStart and self.time <= self._tEnd[self.videoNumber] : #If the cotton was added, and if the video is not finished
                    
                    self.RunSegmentation(); #Runs the segmentation on the ROI
                
                elif self.time < self._tStart : #If the cotton was not added yet
                    
                    self.RunSegmentationBackground(); #Waits... Or displays camera feed if precised
                    
            elif self.videoNumber != 0 : #If the one of the next videos is being processed
                
                if self.time <= self._tEnd[self.videoNumber] : #If the video is not finished
                    
                    self.RunSegmentation(); #Runs the segmentation on the ROI
                    
            if self._args["saveStream"] and self.frameNumber % 30 == 0 :
            
                self.trackingStream.append(self.hStack);
                    
    def RunSegmentation(self) :
        
        #Getting basic image informations
        #----------------------------------------------------------------------------------------------------------------------------------
        
        self.GetFrameInfo();
        
        #Selects only the ROI part of the image for future analysis
        #----------------------------------------------------------------------------------------------------------------------------------
        
        self.distanceRatio = abs(self.upLeftX-self.lowRightX)/self._args["cageLength"]; #Defines the resizing factor for the cage
        
        self.croppedFrame = self.RGBFrame[self.upLeftY:self.lowRightY,self.upLeftX:self.lowRightX]; #Crops the initial frame to the ROI
        
        self.maskDisplay = cv2.cvtColor(self.croppedFrame, cv2.COLOR_BGR2RGB); #[DISPLAY ONLY] Changes the croppedFrame to RGB for display purposes
        self.colorFrame = cv2.cvtColor(self.cloneFrame, cv2.COLOR_BGR2RGB); #[DISPLAY ONLY] Changes the Frame to RGB for display purposes
        
        #Filtering the ROI from noise
        #----------------------------------------------------------------------------------------------------------------------------------
        
        self.hsvFrame = cv2.cvtColor(self.croppedFrame, cv2.COLOR_BGR2HSV); #Changes the croppedFrame LUT to HSV for segmentation
        self.blur = cv2.blur(self.hsvFrame,(5,5)); #Applies a Gaussian Blur to smoothen the image
        self.mask = cv2.inRange(self.blur, self._args["TRESH_MIN"], self._args["TRESH_MAX"]); #Thresholds the image to binary
        self.opening = cv2.morphologyEx(self.mask,cv2.MORPH_OPEN,self._args["kernel"], iterations = 1); #Applies opening operation to the mask for dot removal
        self.closing = cv2.morphologyEx(self.opening,cv2.MORPH_CLOSE,self._args["kernel"], iterations = 1); #Applies closing operation to the mask for large object filling
        self.cnts = cv2.findContours(self.closing.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]; #Finds the contours of the image to identify the meaningful object
        
        #Finding the Mouse
        #----------------------------------------------------------------------------------------------------------------------------------
        
        if self.cnts != [] : #If a contour is found in the binary mask 
            
            self.biggestContour = max(self.cnts, key=cv2.contourArea); #Finds the biggest contour of the binary mask
            self.area = cv2.contourArea(self.biggestContour); #Computes the area of the biggest object from the binary mask
            
            if self.area > self._args["minAreaMask"] : #Future computation is done only if the area of the detected object is meaningful
                
                ((self.x,self.y), self.radius) = cv2.minEnclosingCircle(self.biggestContour);
                self.M = cv2.moments(self.biggestContour); #Computes the Moments of the detected object
                self.center = (int(self.M["m10"] / self.M["m00"]), int(self.M["m01"] / self.M["m00"])); #Computes the Centroid of the detected object
                
                self.StorePosition(); #Stores the position and area of the detected object
                
            else : #If the area of the detected object is too small...
                
                self.Error(); #Specify that no object was detected
                self.StorePosition(); #Stores old position and area
                
        else : #If no contour was detected...
            
            self.Error(); #Specify that no object was detected
            self.area = 0; #Resets the area size to 0
            self.StorePosition(); #Stores old position and area
        
        #Displaying the tracking
        #----------------------------------------------------------------------------------------------------------------------------------
        
        if self._args["showStream"] or self._args["saveStream"] : #If the user specified to show the stream
            
            self.CreateDisplay(); #Creates the montage to be displayed
            
    def RunSegmentationBackground(self) :
        
        if self._args["showStream"] or self._args["saveStream"] :
            
            self.GetFrameInfo();
            
            self.metaDataDisplay = np.zeros((self._H,200,3), np.uint8); #Creates a canvas to write useful info
            
            self.colorFrame = cv2.cvtColor(self.cloneFrame, cv2.COLOR_BGR2RGB); #[DISPLAY ONLY] Changes the Frame to RGB for display purposes
            
            cv2.putText(self.metaDataDisplay,"Waiting...",(8,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1); #Displays waiting...
            cv2.rectangle(self.colorFrame, (self.upLeftX,self.upLeftY), (self.lowRightX,self.lowRightY),(255,0,0), 1); #Displays the ROI square on the image
            
            self.hStack = np.hstack((self.metaDataDisplay, self.colorFrame)); #Generates a horizontal stacked image with useful info and stream
            
    def GetFrameInfo(self) :
        
        self._H,self._W,self._C = self.RGBFrame.shape; #Gets the Height, Width and Color of each frame
        self.cloneFrame = self.RGBFrame.copy(); #[DISPLAY ONLY] Creates a clone frame for display purposes
        
        self.upLeftX = int(self._refPt[0][0]); #Defines the Up Left ROI corner X coordinates
        self.upLeftY = int(self._refPt[0][1]); #Defines the Up Left ROI corner Y coordinates
        self.lowRightX = int(self._refPt[1][0]); #Defines the Low Right ROI corner X coordinates
        self.lowRightY = int(self._refPt[1][1]); #Defines the Low Right ROI corner Y coordinates
            
    def CreateDisplay(self) :
        
        self.metaDataDisplay = np.zeros((self._H,200,3), np.uint8); #Creates a canvas to write useful info
        
        self.ComputeDistanceTraveled();
        
        #----------------------------------------------------------------------------------------------------------------------------------
        #Draw text, contours, mouse tracker on the frames
        
        try :
            
            cv2.drawContours(self.maskDisplay, [self.biggestContour], 0, (0,255,0), 1); #Draws the contour of the detected object on the image
            cv2.putText(self.metaDataDisplay,'Detection : '+str("Successful"),(8,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1); #Displays the detection status
            
        except :
            
            cv2.putText(self.metaDataDisplay,'Detection : '+str("Failed"),(8,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1); #Displays the detection status
            
        self.colorFrame[self.upLeftY:self.upLeftY+self.maskDisplay.shape[0],\
                        self.upLeftX:self.upLeftX+self.maskDisplay.shape[1]] = self.maskDisplay; 
                        
        cv2.circle(self.colorFrame, self.correctedCenter, 1, (0, 0, 255), 2); #Draws a the object Centroid as a point
 
        cv2.putText(self.metaDataDisplay,'mask size : '+str(int(self.area))+'px2',(8,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1); #Displays the mask size in real-time
        cv2.putText(self.metaDataDisplay,'dist : '+str("%.2f" % self._distance)+'cm',(8,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1); #Displays the distance traveled by the object in real-time
        cv2.putText(self.metaDataDisplay,'speed : '+str("%.2f" % self.realTimeSpeed)+'cm/s',(8,45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1); #Displays the mask size in real-time
        cv2.rectangle(self.colorFrame, (self.upLeftX,self.upLeftY), (self.lowRightX,self.lowRightY),(255,0,0), 1); #Displays the ROI square on the image
        
        self.hStack = np.hstack((self.metaDataDisplay, self.colorFrame)); #Generates a horizontal stacked image with useful info and stream
        
    def ComputeDistanceTraveled(self) :
        
        if len(self.realTimePosition) == 2 and not None in self.realTimePosition : #Runs only if two positions are available and if previous or current position are not None
                
                self.realTimeSpeed = sqrt((self.realTimePosition[1][0]-self.realTimePosition[0][0])**2+\
                                      (self.realTimePosition[1][1]-self.realTimePosition[0][1])**2)/self.distanceRatio; #Computes distance
    
                if self.realTimeSpeed >= self._args["minDist"] : #If the distance is higher than the minimal distance (filtering for movement noise)
                    
                    self._distance += (self.realTimeSpeed); #Adds the value to the cumulative distance varaible
                    
    def UpdatePosition(self) :

        if len(self.realTimePosition) == 0 :
            self.realTimePosition.append(self.center);
        elif len(self.realTimePosition) == 1 :
            self.realTimePosition.append(self.center);
        elif len(self.realTimePosition) == 2 :
            self.realTimePosition[0] = self.realTimePosition[1];
            self.realTimePosition[1] = self.center;
            
    def StorePosition(self) :
        
        self._positions.append(self.center);
        self._maskAreas.append(self.area);
        
        if self.center != None :
            self.UpdatePosition();
            self.correctedCenter = (self.center[0]+self.upLeftX,\
                                    self.center[1]+self.upLeftY);
            
    def Error(self) : 
        
        self._errors += 1;
        
        if self._args["showStream"] :
            
            if self._errors == 1 :
                
                print("[WARNING] No contour detected, assuming old position !");
                
            elif self._errors % 100 == 0 :
                
                print("[WARNING] No contour detected, assuming old position !");
                self._errors = 0;
              
    def ReturnTracking(self) :
        
        if self._args["showStream"] :
            return self.hStack;
        else :
            pass;
    
    
class DisplaySegmentation() :
    
    def __init__(self,mice,mode="v") :
        
        if len(mice) == 2 :
            
            if mode == "v" :
        
                self.Stack = np.vstack((mice[1],mice[0]));
                cv2.imshow('frame',self.Stack);
                
            elif mode == "h" :
                
                self.Stack = np.hstack((mice[0],mice[1]));
                cv2.imshow('frame',self.Stack);
            
        elif len(mice) == 1 :
            cv2.imshow('frame',mice[0]);
   
         
def TopTracker(data,**kwargs) :
    
    Tracker = data; #Initialize all variables

    for capture in kwargs["captures"] :
    
        #Starts segmentation#
        while(True):
            
            #trackers = [];
            
            Tracker.OffLineTracker();
            tracker = Tracker.ReturnTracking();
            #trackers.append(tracker);
            
            #Runs only if the videoLength is not null
            if not Tracker._tEnd[Tracker.videoNumber] == 0 :
                
                #----------------------------------------------------------------------------------------------------------------------------------
                #Runs only every 10 minutes of the video being analyzed
                if Tracker.frameNumber%(600*kwargs["framerate"]) == 0 :
                    
                    print(len(Tracker._positions))
                    utils.PrintColoredMessage('Loaded and analyzed : '+str(Tracker.frameNumber)+'/'+str(int(Tracker._tEnd[Tracker.videoNumber]*kwargs["framerate"]))+\
                        ' = '+(str(int(float(Tracker.frameNumber)/float(Tracker._tEnd[Tracker.videoNumber]*kwargs["framerate"])*100)))\
                        +'% frames from video n°'+str(Tracker.videoNumber)+'/'+str(Tracker._nVideos), "darkgreen");
                    utils.PrintColoredMessage(utils.PrintLoadingBar(int(float(Tracker.frameNumber)/float(Tracker._tEnd[Tracker.videoNumber]*kwargs["framerate"])*100)),"darkgreen");
                        
                #----------------------------------------------------------------------------------------------------------------------------------
                if Tracker.frameNumber == int(Tracker._tEnd[Tracker.videoNumber]*kwargs["framerate"]) :
        
                    print('\n'); 
                    utils.PrintColoredMessage('#########################################################',"darkgreen");
                    utils.PrintColoredMessage("[INFO] Video {0} for mouse {1} has been successfully analyzed".format(str(Tracker.videoNumber),Tracker._mouse),"darkgreen");
                    utils.PrintColoredMessage('#########################################################',"darkgreen");
                    utils.PlaySound(2,params.sounds['Purr']); #Plays sound when code finishes
                    Tracker.videoNumber += 1; #Increments videoNumber variable to keep track which video is being processed
                    Tracker.frameNumber = 0;
                    break;

            if kwargs["showStream"] :
                DisplaySegmentation(tracker,mode="v");
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break;
                        
            

def SaveTracking(data,directory):
    
    savingDir = directory+'Results';
    utils.CheckDirectoryExists(savingDir);
        
    first = 0;

    for point in data._positions :
        if point == None :
            first += 1;
    
    for n,point in enumerate(data._positions) :
        if point == None :
            data._positions[n] = data._positions[first];
            
    np.save(os.path.join(savingDir,'Mouse_Data_All_'+str(data._mouse)+'_Points.npy'),data._positions);
    np.save(os.path.join(savingDir,'Mouse_Data_All_'+str(data._mouse)+'_refPt.npy'),data._refPt);
    np.save(os.path.join(savingDir,'Mouse_Data_All_'+str(data._mouse)+'_Areas.npy'),data._maskAreas);
    
def SaveStream(data,directory):
    
    savingDir = directory+'Results';
    utils.CheckDirectoryExists(savingDir);
    
    name = 'Tracking_{0}.avi'.format(data._mouse);

    images = data.trackingStream;
    frame = images[0];
    _H, _W, _L = frame.shape;
    
    video = cv2.VideoWriter(os.path.join(savingDir,name), 0, 1, (_W,_H));
    
    for image in images:
        video.write(image);
    
    cv2.destroyAllWindows();
    video.release();

                