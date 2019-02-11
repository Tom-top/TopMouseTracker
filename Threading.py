#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:19:06 2019

@author: tomtop
"""

import numpy as np;
import cv2;
import time;
import os;
import xlwt;
from math import pi,tan;
from threading import Thread;

import TopMouseTracker.Utilities as utils;

class Kinect() :
    
    def __init__(self,**kwargs) : 
        
        self._args = kwargs;
        self.gridRes = self._args["gridRes"];
        self.kinectHeight = 60; #cm
        self.kinectHorizontalAngle = 70.6; #degree
        self.kinectVerticalAngle = 60; #degree
        self.topBoxSize = 50; #cm
        self.fps = [];
        
    def GetFrame(self,camera,which,resize) :
        
        if which == "rgb" :
            
            frame = camera.get_last_color_frame();
            W,H = camera.color_frame_desc.Width,camera.color_frame_desc.Height;
            frame = frame.reshape(H,W,-1).astype(np.uint8);
            frame = cv2.resize(frame, (int(W/resize),int(H/resize)));
            return frame;
            
        elif which == "depth" :
            
            frame = camera.get_last_depth_frame();
            W,H = camera.depth_frame_desc.Width,camera.depth_frame_desc.Height;
            frame8bit = frame.reshape(H,W,-1).astype(np.uint8);
            frame8bit = cv2.resize(frame8bit, (int(W/resize),int(H/resize)));
            frame16bit = frame.reshape(H,W,-1).astype(np.uint16);
            frame16bit = cv2.resize(frame16bit, (int(W/resize),int(H/resize)));
            return frame8bit,frame16bit;

    def LoadRGBDEPTH(self,resizeRGB,resizeDEPTH) :
        
        RGBFrame = self.GetFrame(self._args["kinectRGB"],"rgb",resizeRGB);
        RGBFrame = cv2.cvtColor(RGBFrame, cv2.COLOR_BGR2RGB); 
        RGBFrame = cv2.cvtColor(RGBFrame, cv2.COLOR_RGB2BGR);
        RGBFrame = cv2.flip(RGBFrame,0);
        
        DEPTHFrame8bit,DEPTHFrame16bit = self.GetFrame(self._args["kinectDEPTH"],"depth",resizeDEPTH);
        
        DEPTHFrame8bit = cv2.cvtColor(DEPTHFrame8bit, cv2.COLOR_GRAY2BGR); 
        DEPTHFrame8bit = cv2.flip(DEPTHFrame8bit,0);
        
        #DEPTHFrame16bit = cv2.cvtColor(DEPTHFrame16bit, cv2.COLOR_GRAY2BGR); 
        #DEPTHFrame16bit = cv2.flip(DEPTHFrame16bit,0);
        
        return RGBFrame,DEPTHFrame8bit#,DEPTHFrame16bit;
    
    def CreateDisplay(self) :
        
        RGBFrame,DEPTHFrame8bit = self.LoadRGBDEPTH(3,1);
        
        H,W,_ = RGBFrame.shape;
        DEPTHFrame8bit = cv2.resize(DEPTHFrame8bit, (W,H));
        hStack = np.hstack((RGBFrame, DEPTHFrame8bit));
        
        return hStack;
    
    def TestKinect(self,grid=True) : 
        
        while True :
            
            RGBFrame,DEPTHFrame8bit = self.LoadRGBDEPTH(2,1);
            
            clone = DEPTHFrame8bit.copy()
            
            if grid : 

                box_size = 50;
                height,width,depth = DEPTHFrame8bit.shape;
                
                horizontal_length = 2*( tan( (self.kinectHorizontalAngle*(pi/180)) /2 )*self.kinectHeight );
                vertical_length = 2*( tan( (self.kinectVerticalAngle*(pi/180)) /2 )*self.kinectHeight );
                
                ratio1 = width/horizontal_length;
                ratio2 = height/vertical_length;
                
                average_ratio = int((ratio1+ratio2)/2);
                
                shift = 0;
                
                horizontal_position = ( width-(box_size*average_ratio) )/2+shift;
                vertical_position = ( height-(box_size*average_ratio) )/2;
                
                cv2.rectangle(DEPTHFrame8bit, (int(horizontal_position), int(vertical_position)),\
                              (int(horizontal_position+box_size*average_ratio), int(vertical_position+box_size*average_ratio)), (255,0,0), 1);
                              
                gridres = self.gridRes;      
                        
                for x in np.arange(int(horizontal_position)+gridres,int(horizontal_position)+(box_size*average_ratio),gridres) :
                    
                    for y in np.arange(int(vertical_position)+gridres,int(vertical_position)+(box_size*average_ratio),gridres) : 
                        
                        local_pixel_values = [];
                        
                        for x_pixel in np.arange(-gridres/2,gridres/2) :
                            for y_pixel in np.arange(-gridres/2,gridres/2) :
                                pixel_value = clone[int(y+y_pixel)][int(x+x_pixel)][0];
                                local_pixel_values.append(pixel_value);
                                
                        avg_local_value = sum(local_pixel_values)/len(local_pixel_values);
                        
                        if avg_local_value > self._args["depthMaxThresh"] : 
                            cv2.circle(DEPTHFrame8bit,(x,y), gridres-12, (255,0,0), 1);
                        elif avg_local_value < self._args["depthMinThresh"] : 
                            cv2.circle(DEPTHFrame8bit,(x,y), gridres-12, (0,0,255), 1);
                        else : 
                            cv2.circle(DEPTHFrame8bit,(x,y), gridres-12, (0,255,0), 1);
                        
                        cv2.putText(DEPTHFrame8bit, str(int(avg_local_value)), (x-10,y+2),cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255));
            
            cv2.imshow("RGBframe",RGBFrame); 
            cv2.imshow("DEPTHframe",DEPTHFrame8bit); 
   
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break;
                
        cv2.destroyAllWindows();
        
    def startThread(self) :
        
        Thread(target=self.update, args=()).start();
        return self;
    
    def update(self) :
        
        while True :
            
            self.FrameRGB,self.FrameDEPTH8Bit = self.LoadRGBDEPTH(1,1);
            self.frameCnt += 1;
            
            self.tNow = time.time();
            
            if self.display :
        
                self.downSampledRGB = cv2.resize(self.FrameRGB, (0,0), fx=0.3, fy=0.3);
                cv2.imshow('RGB',self.downSampledRGB);
                cv2.imshow('DEPTH',self.FrameDEPTH8Bit);
            
            if self.tStart-self.tNow > 1*60 : #Starts saving the frames after 1 minute of sampling
        
                if self.trigger :
                    
                    self.sampledFrameRate = self.frameCnt/(self.tStart-self.tNow);
                    
                    self.RGBWriter = cv2.VideoWriter(os.path.join(self.dataDir,\
                                            '{0}_{1}-{2}-{3}_{4}-{5}-{6}.avi'.format(*self.RGBString)),\
                                            self._args["fourcc"],self.sampledFrameRate,(self.wRGB,self.hRGB));
                                                
                    self.DEPTH8BitWriter = cv2.VideoWriter(os.path.join(self.dataDir,\
                                            '{0}_{1}-{2}-{3}_{4}-{5}-{6}.avi'.format(*self.DEPTH8BitString)),\
                                            self._args["fourcc"],self.sampledFrameRate,(self.wDEPTH,self.hDEPTH));
                                                      
                    self.trigger = False;
            
                self.RGBWriter.write(self.FrameRGB);
                self.DEPTHWriter.write(self.FrameDEPTH8Bit);
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break;

    def PlayAndSave(self,display=True) :
        
        self.display = display;
        self.trigger = True;
        
        self.time = time.localtime(time.time());
        self.tStart = time.time();
        self.frameCnt = 0;
        
        self.dataDirName = '{0}-{1}-{2}_{3}-{4}-{5}'.format(self.time.tm_mday,\
                        self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                        self.time.tm_min,self.time.tm_sec);
                            
        self.dataDir = os.path.join(self._args["savingDir"],self.dataDirName);
        utils.CheckDirectoryExists(self.dataDir);
        
        testFrameRGB = self.GetFrame(self._args["kinectRGB"],"rgb",1);
        
        try :
            self.hRGB,self.wRGB,_ = testFrameRGB.shape
        except : 
            self.hRGB,self.wRGB = testFrameRGB.shape
        
        testFrameDEPTH,_ = self.GetFrame(self._args["kinectDEPTH"],"depth",1);
        
        try :
            self.hDEPTH,self.wDEPTH,_ = testFrameDEPTH.shape
        except :
            self.hDEPTH,self.wDEPTH = testFrameDEPTH.shape
        
        self.RGBString = self._args["rawVideoFileName"],self.time.tm_mday,\
                        self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                        self.time.tm_min,self.time.tm_sec;
    
        self.DEPTH8BitString = self._args["depthVideoFileName8Bit"],self.time.tm_mday,\
                        self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                        self.time.tm_min,self.time.tm_sec;
    
        self.startThread();
        
        self.RGBWriter.release();
        self.DEPTH8BitWriter.release();
        
        self.tEnd = time.time();
        self.frameRate = self.frameCnt/(self.tStart-self.tEnd);
        
        self.saveMetaData();
                
        cv2.destroyAllWindows();
        
    def saveMetaData(self) :
        
        self.metaDataString = "MetaData",self.time.tm_mday,\
                        self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                        self.time.tm_min,self.time.tm_sec;
                        
        self.metaDataFile = os.path.join(self.dataDir,'{0}_{1}-{2}-{3}_{4}-{5}-{6}.xls'.format(*self.metaDataString));
        self.metaData = xlwt.Workbook();
        sheet = self.metaData.add_sheet("MetaData");
        
        sheet.write(0, 0, "Mice");
        sheet.write(0, 1, self._args["mice"]);
        
        sheet.write(1, 0, "Time_Stamp");
        sheet.write(1, 1, "{0}-{1}-{2}_{3}-{4}-{5}".format(self.time.tm_mday,\
                        self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                        self.time.tm_min,self.time.tm_sec));
        
        sheet.write(2, 0, "Elapsed_Time");
        sheet.write(2, 1, self.tEnd-self.tStart);
        
        sheet.write(3, 0, "Sampled_Framerate");
        sheet.write(3, 1, self.sampledFrameRate);
        
        sheet.write(4, 0, "Real_Framerate");
        sheet.write(4, 1, self.frameRate);
        
        sheet.write(5, 0, "nFrames");
        sheet.write(5, 1, self.frameCnt);
        
        sheet.write(6, 0, "Fourcc");
        sheet.write(6, 1, self._args["fourcc"]);
        
        self.metaData.save(self.metaDataFile);