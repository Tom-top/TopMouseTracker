#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:03:44 2018

@author: tomtop
"""

import os;
import cv2;
import time;
import numpy as np;
from math import sqrt;

import TopMouseTracker.IO as IO;

def getColorFrame(stream,resize):
    
    _frameRGB = stream.get_last_color_frame();
    
    _W,_H = stream.color_frame_desc.Width,stream.color_frame_desc.Height;
    _frameRGB = _frameRGB.reshape(_H,_W,-1).astype(np.uint8);
    _frameRGB = cv2.resize(_frameRGB, (int(_W/resize),int(_H/resize)));
    
    return _frameRGB;

def getDepthFrame(stream,resize):
    
    _frameDEPTH = stream.get_last_depth_frame();
    
    W,H = stream.depth_frame_desc.Width,stream.depth_frame_desc.Height
    _frameDEPTH = _frameDEPTH.reshape((H, W,-1)).astype(np.uint16)
    _frameDEPTH = cv2.resize(_frameDEPTH, (int(W/resize),int(H/resize)))
    
    return _frameDEPTH;


class Tracker():
    
    '''Class that holds all parameters usefull for the segmentation
    
    Params :
        frame (array) : testFrame coming from the VideoLoader function
        data (list) : List holding all the metadata for each mouse coming from
        the excel spreadsheet
    '''
    
    def __init__(self,mode="light",**kwargs) :
        
        #General variables
        #----------------------------------------------------------------------
        self.mode = mode; #Mode argument to launch the regular/night mode tracker
        self._args = kwargs; #Arguments
        self._mouse = self._args["segmentation"]["mouse"]; #Name of the mouse
        self._H, self._W = self._args["main"]["rawFrameRGB"].shape

        #Global tracking variables
        #----------------------------------------------------------------------
        self._positions = []; #Position of the mouse every s/frameRate
        self._maskAreas = []; #Size of the mouse mask every s/frameRate
        self._distance = 0.; #Cumulative distance traveled by the mouse
        self._errors = 0; #Counter for the number of times that the trackers fails to segment the animal
        
        #Real-Time tracking variables
        #----------------------------------------------------------------------
        self.realTimePosition = []; #Position of the mouse in real time
        self.realTimeSpeed = 0.; #Speed of the mouse in real time
        self.center = None; #Centroid (x,y) of the mouse binary mask in real time
        self.correctedCenter = None; #Corrected centroid (x,y) of the mouse binary mask in real time
        self.trackingStream = [];
        
        #Saving variables
        #----------------------------------------------------------------------
        self.time = time.localtime(time.time());
        
    def SetROI(self) :
        
        if self.mode == "light" :
        
            self._refPt = IO.CroppingROI(self._args["main"]["testFrameRGB"][0].copy()).roi(); #Defining the ROI for segmentation
            
        if self.mode == "dark" :
            
            self._refPt = IO.CroppingROI(self._args["main"]["testFrameDEPTH"][0].copy()).roi(); #Defining the ROI for segmentation
        
    def Main(self):
        
        if self.mode == "light" : #If the regular mode was choosen
        
            #Get frame from stream
            #----------------------------------------------------------------------
            self.RGBFrame = getColorFrame(self._args["main"]["kinectRGB"],1);
               
            #If the RGB kinect is working
            #----------------------------------------------------------------------
            
            self.RunSegmentationMouse(self.RGBFrame,self._args["segmentation"]["threshMinRGB"],self._args["segmentation"]["threshMaxRGB"]); #Runs the segmentation on the ROI
            
            #Displaying the tracking
            #----------------------------------------------------------------------
            
            if self._args["display"]["showStream"] or self._args["saving"]["saveVideo"] : #If the user specified to show the stream
                
                self.CreateDisplay(); #Creates the montage to be displayed
                
        if self.mode == "dark" : #If the regular mode was choosen
            
            #Get frame from stream
            #----------------------------------------------------------------------
            self.DEPTHFrame = getDepthFrame(self._args["main"]["kinectDEPTH"],1);
            
            #If the DEPTH kinect is working
            #----------------------------------------------------------------------
            
            self.RunSegmentationMouse(self.DEPTHFrame,self._args["segmentation"]["threshMinDEPTH"],self._args["segmentation"]["threshMaxDEPTH"]); #Runs the segmentation on the ROI
                    
    def RunSegmentationMouse(self,frame,threshMin,threshMax) :
        
        #Getting basic image informations
        #----------------------------------------------------------------------------------------------------------------------------------
        
        self.GetFrameInfo(frame);
        
        #Selects only the ROI part of the image for future analysis
        #----------------------------------------------------------------------------------------------------------------------------------
        
        self.distanceRatio = abs(self.upLeftX-self.lowRightX)/self._args["segmentation"]["cageLength"]; #Defines the resizing factor for the cage
        
        self.croppedFrame = frame[self.upLeftY:self.lowRightY,self.upLeftX:self.lowRightX]; #Crops the initial frame to the ROI
        
        if self.mode == "light" :
        
            self.maskDisplay = cv2.cvtColor(self.croppedFrame, cv2.COLOR_BGR2RGB); 
            self.maskDisplay = cv2.cvtColor(self.maskDisplay, cv2.COLOR_RGB2BGR); #[DISPLAY ONLY] Changes the croppedFrame to RGB for display purposes
            
            self.colorFrame = cv2.cvtColor(self.cloneFrame, cv2.COLOR_BGR2RGB); 
            self.colorFrame = cv2.cvtColor(self.colorFrame, cv2.COLOR_RGB2BGR); #[DISPLAY ONLY] Changes the Frame to RGB for display purposes
            
        elif self.mode == "dark" :
            
            self.maskDisplay = cv2.cvtColor(self.croppedFrame, cv2.COLOR_BGR2RGB);       
            self.colorFrame = cv2.cvtColor(self.cloneFrame, cv2.COLOR_BGR2RGB); 
        
        #Filtering the ROI from noise
        #----------------------------------------------------------------------------------------------------------------------------------
        
        self.blur = cv2.blur(self.hsvFrame,(5,5)); #Applies a Gaussian Blur to smoothen the image
        self.mask = cv2.inRange(self.blur, threshMin, threshMax); #Thresholds the image to binary
        self.opening = cv2.morphologyEx(self.mask,cv2.MORPH_OPEN,self._args["segmentation"]["kernel"], iterations = 1); #Applies opening operation to the mask for dot removal
        self.closing = cv2.morphologyEx(self.opening,cv2.MORPH_CLOSE,self._args["segmentation"]["kernel"], iterations = 1); #Applies closing operation to the mask for large object filling
        self.cnts = cv2.findContours(self.closing.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]; #Finds the contours of the image to identify the meaningful object
        
        #Finding the Mouse
        #----------------------------------------------------------------------------------------------------------------------------------
        
        if self.cnts != [] : #If a contour is found in the binary mask 
            
            self.biggestContour = max(self.cnts, key=cv2.contourArea); #Finds the biggest contour of the binary mask
            self.area = cv2.contourArea(self.biggestContour); #Computes the area of the biggest object from the binary mask
            
            if self.area > self._args["segmentation"]["minAreaMask"] and self.area < self._args["segmentation"]["maxAreaMask"] : #Future computation is done only if the area of the detected object is meaningful
                
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
            
    def GetFrameInfo(self,frame) :
        
        self._H,self._W,self._C = frame.shape; #Gets the Height, Width and Color of each frame
        self.cloneFrame = frame.copy(); #[DISPLAY ONLY] Creates a clone frame for display purposes
        
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
            
            cv2.drawContours(self.maskDisplay, [self.biggestContour], 0, (0,0,255), 1); #Draws the contour of the detected object on the image
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
    
                if self.realTimeSpeed >= self._args["segmentation"]["minDist"] : #If the distance is higher than the minimal distance (filtering for movement noise)
                    
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
        
        if self._args["display"]["showStream"] :
            
            if self._errors == 1 :
                
                print("[WARNING] No contour detected, assuming old position !");
                
            elif self._errors % 100 == 0 :
                
                print("[WARNING] No contour detected, assuming old position !");
                self._errors = 0;
              
    def ReturnTracking(self) :
        
        if self._args["display"]["showStream"] or self._args["saving"]["saveVideo"] :
                
            return self.RGBFrame,self.hStack;
        
        else :
            
            pass;

    
class DisplaySegmentation() :
    
    def __init__(self,trackers,**kwargs) :
        
        self._args = kwargs;
        
        if len(trackers) == 2 :
            
            if self._args["mode"] == "v" :
        
                self.Stack = np.vstack((trackers[1],trackers[0]));
                cv2.imshow('Tracking',self.Stack);
                
            elif self._args["mode"] == "h" :
                
                self.Stack = np.hstack((trackers[0],trackers[1]));
                cv2.imshow('Tracking',self.Stack);
            
        elif len(trackers) == 1 :
            
            self.Stack = trackers[0];
            cv2.imshow('Tracking',self.Stack);
            
    def ReturnStack(self) :
            
        return self.Stack;
    
    
class SavingSegmentation() :
    
    def __init__(self,**kwargs) :
        
        self._args = kwargs;
            
        self.rawString = self._args["saving"]["rawVideoFileName"],self.time.tm_mday,\
                    self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                    self.time.tm_min,self.time.tm_sec;
                    
        self.rawWriter = self._args["saving"]["rawWriter"];
        
        self.rawWriter = cv2.VideoWriter(os.path.join(self._args["main"]["resultDir"],\
                                            '{0}_{1}-{2}-{3}_{4}-{5}-{6}.avi'.format(* self.rawString)),\
                                            self._args["saving"]["fourcc"],self._args["saving"]["framerate"],(self._W,self._H,));
    
        self.segString = self._args["saving"]["segVideoFileName"],self.time.tm_mday,\
                        self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                        self.time.tm_min,self.time.tm_sec;
                        
        self.segWriter = self._args["saving"]["segWriter"];
        
        if len(self._args["main"]["mice"]) == 2 and self._args["saving"]["mode"] == "v" :
        
            self.segW = self._W+200;
            self.segH = 2*self._H;
            
        if len(self._args["main"]["mice"]) == 2 and self._args["saving"]["mode"] == "h" :
        
            self.segW = 2*self._W+400;
            self.segH = self._H;
            
        if len(self._args["main"]["mice"]) == 1 :
        
            self.segW = self._W;
            self.segH = self._H;
                        
        self.segWriter = cv2.VideoWriter(os.path.join(self._args["main"]["resultDir"],\
                                            '{0}_{1}-{2}-{3}_{4}-{5}-{6}.avi'.format(*self.segString)),\
                                            self._args["saving"]["fourcc"],self._args["saving"]["framerate"],(self.segW,self.segH));
                
    def Save(self,rawImage,segImage) :
        
        self.rawImage = rawImage;
        self.segImage = segImage;
        
        self.rawWriter.write(self.rawImage);
        self.segWriter.write(self.segImage);
        
    def Release(self) :
        
        self.rawWriter.release();
        self.segWriter.release(); 
        
def SaveTracking(trackers,**kwargs) :
    
    args = kwargs;
    
    for k,v in trackers.items() :
        
        first = 0;
    
        for point in v._positions :
            if point == None :
                first += 1;
        
        for n,point in enumerate(v._positions) :
            if point == None :
                v._positions[n] = v._positions[first];
                
        np.save(os.path.join(args["main"]["resultDir"],'Mouse_Data_All_'+str(v._mouse)+'_Points.npy'),v._positions);
        np.save(os.path.join(args["main"]["resultDir"],'Mouse_Data_All_'+str(v._mouse)+'_refPt.npy'),v._refPt);
        np.save(os.path.join(args["main"]["resultDir"],'Mouse_Data_All_'+str(v._mouse)+'_Areas.npy'),v._maskAreas);
    
def KinectTopMouseTracker(trackers,**kwargs) :
    
    args = kwargs;
    writers = SavingSegmentation();
    
    #Starts segmentation#
    while(True):
        
        segmentation = [];
        
        for k,v in trackers.items() :
           seg,raw = v.ReturnTracking();
           segmentation.append(seg);
        
        if args["display"]["showStream"] :
            display = DisplaySegmentation(segmentation)
            montage = display.ReturnStack();
        
        if args["saving"]["saveVideo"] :
            writers.Save(raw,montage);
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
            
    if args["saving"]["saveVideo"] :
        writers.Release();
        
    cv2.destroyAllWindows();
    
    SaveTracking(trackers,args);

                