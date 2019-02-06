#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 11:31:48 2019

@author: ADMIN
"""

import numpy as np;
import cv2;
import time;
import os;
from math import pi,tan;

class Kinect() :
    
    def __init__(self,**kwargs) : 
        
        self._args = kwargs;
        self.gridRes = self._args["gridRes"];
        self.kinectHeight = 60; #cm
        self.kinectHorizontalAngle = 70.6; #degree
        self.kinectVerticalAngle = 60; #degree
        self.topBoxSize = 50; #cm
        
    def GetFrame(self,camera,which,resize) :
        
        if which == "rgb" :
            
            frame = camera.get_last_color_frame();
            W,H = camera.color_frame_desc.Width,camera.color_frame_desc.Height;
            frame = frame.reshape(H,W,-1).astype(np.uint8);
            
        elif which == "depth" :
            
            frame = camera.get_last_depth_frame();
            W,H = camera.depth_frame_desc.Width,camera.depth_frame_desc.Height;
            frame = frame.reshape(H,W,-1).astype(np.uint16);
            
        frame = cv2.resize(frame, (int(W/resize),int(H/resize)));
        
        return frame;
    
    def LoadRGBDEPTH(self,resizeRGB,resizeDEPTH) :
        
        RGBFrame = self.GetFrame(self._args["kinectRGB"],"rgb",resizeRGB);
        RGBFrame = cv2.cvtColor(RGBFrame, cv2.COLOR_BGR2RGB); 
        RGBFrame = cv2.cvtColor(RGBFrame, cv2.COLOR_RGB2BGR);
        RGBFrame = cv2.flip(RGBFrame,0);
        
        DEPTHFrame = self.GetFrame(self._args["kinectDEPTH"],"depth",resizeDEPTH);
        DEPTHFrame = cv2.cvtColor(DEPTHFrame, cv2.COLOR_GRAY2BGR); 
        DEPTHFrame = cv2.flip(DEPTHFrame,0);
        
        return RGBFrame,DEPTHFrame;
    
    def CreateDisplay(self) :
        
        RGBFrame,DEPTHFrame = self.LoadRGBDEPTH(2,1/2);
        
        H,W = RGBFrame.shape;
        DEPTHFrame = cv2.resize(DEPTHFrame, (W,H));
        hStack = np.hstack(RGBFrame, DEPTHFrame);
        
        return hStack;
        
    def TestKinect(self) :
        
        while True :
            
            #hStack = self.CreateDisplay()
            RGBFrame,DEPTHFrame = self.LoadRGBDEPTH(2,1);
            #cv2.imshow('RGB&DEPTH',hStack);
            
            clone = DEPTHFrame.copy()

            box_size = 50;
            height,width,depth = DEPTHFrame.shape;
            
            horizontal_length = 2*( tan( (self.kinectHorizontalAngle*(pi/180)) /2 )*self.kinectHeight );
            vertical_length = 2*( tan( (self.kinectVerticalAngle*(pi/180)) /2 )*self.kinectHeight );
            
            ratio1 = width/horizontal_length;
            ratio2 = height/vertical_length;
            
            average_ratio = int((ratio1+ratio2)/2);
            
            shift = 0;
            
            horizontal_position = ( width-(box_size*average_ratio) )/2+shift;
            vertical_position = ( height-(box_size*average_ratio) )/2;
            
            cv2.rectangle(DEPTHFrame, (int(horizontal_position), int(vertical_position)),\
                          (int(horizontal_position+box_size*average_ratio), int(vertical_position+box_size*average_ratio)), (255,0,0), 1);
                          
            gridres = self.gridRes;      
                    
            for x in np.arange(int(horizontal_position)+gridres,int(horizontal_position)+(box_size*average_ratio),gridres) :
                
                for y in np.arange(int(vertical_position)+gridres,int(vertical_position)+(box_size*average_ratio),gridres) : 
                    
                    #cv2.circle(frame,(x,y), 1, (0,255,0), 1);
                    
                    local_pixel_values = [];
                    
                    for x_pixel in np.arange(-gridres/2,gridres/2) :
                        for y_pixel in np.arange(-gridres/2,gridres/2) :
                            #cv2.circle(frame,(int(x+x_pixel),int(y+y_pixel)), 1, (0,255,0), 1);
                            pixel_value = clone[int(y+y_pixel)][int(x+x_pixel)][0];
                            local_pixel_values.append(pixel_value);
                            
                    avg_local_value = sum(local_pixel_values)/len(local_pixel_values);
                    
                    if avg_local_value > self._args["depthMaxThresh"] : 
                        cv2.circle(DEPTHFrame,(x,y), gridres-12, (255,0,0), 1);
                    elif avg_local_value < self._args["depthMinThresh"] : 
                        cv2.circle(DEPTHFrame,(x,y), gridres-12, (0,0,255), 1);
                    else : 
                        cv2.circle(DEPTHFrame,(x,y), gridres-12, (0,255,0), 1);
                    
                    cv2.putText(DEPTHFrame, str(int(avg_local_value)), (x-10,y+2),cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255));
            
            cv2.imshow("RGBframe",RGBFrame); 
            cv2.imshow("DEPTHframe",DEPTHFrame);       
   
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break;
                
        cv2.destroyAllWindows();
        
    def PlayAndSave(self,display=True) :
        
        self.time = time.localtime(time.time());
        
        testFrameRGB = self.GetFrame(self._args["kinectRGB"],"rgb",1);
        try :
            hRGB,wRGB,_ = testFrameRGB.shape
        except : 
            hRGB,wRGB = testFrameRGB.shape
        
        testFrameDEPTH = self.GetFrame(self._args["kinectRGB"],"depth",1);
        try :
            hDEPTH,wDEPTH,_ = testFrameDEPTH.shape
        except :
            hDEPTH,wDEPTH = testFrameDEPTH.shape
        
        self.RGBString = self._args["rawVideoFileName"],self.time.tm_mday,\
                        self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                        self.time.tm_min,self.time.tm_sec;
            
        self.RGBWriter = cv2.VideoWriter(os.path.join(self._args["resultDir"],\
                                            '{0}_{1}-{2}-{3}_{4}-{5}-{6}.avi'.format(*self.RGBString)),\
                                            self._args["fourcc"],self._args["framerate"],(wRGB,hRGB));
    
        self.DEPTHString = self._args["depthVideoFileName"],self.time.tm_mday,\
                        self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                        self.time.tm_min,self.time.tm_sec;
            
        self.DEPTHWriter = cv2.VideoWriter(os.path.join(self._args["resultDir"],\
                                            '{0}_{1}-{2}-{3}_{4}-{5}-{6}.avi'.format(*self.DEPTHString)),\
                                            self._args["fourcc"],self._args["framerate"],(wDEPTH,hDEPTH));
    
        while True :
            
            self.FrameRGB,self.FrameDEPTH = self.LoadRGBDEPTH(1,1);
            
            if display :

                hStack = self.CreateDisplay();
                cv2.imshow('RGB&DEPTH',hStack);
            
            self.RGBWriter.write(self.FrameRGB);
            self.DEPTHWriter.write(self.FrameDEPTH);
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break;
                
        self.RGBWriter.release();
        self.DEPTHWriter.release();
                
        cv2.destroyAllWindows();
    
        
        
        
        
        
        
        