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
import xlwt;
from math import pi,tan;

import TopMouseTracker.utilities as utils

class Kinect() :
    
    def __init__(self,**kwargs) : 
        
        self._args = kwargs;
        
        self.kinectHeight = 60; #cm
        self.kinectHorizontalAngle = 70.6; #degree
        self.kinectVerticalAngle = 60; #degree
        self.topBoxSize = 50; #cm
        
    def GetRGBFrame(self, resize) :
        
        self.RGBFrame = self._args["kinectRGB"].get_last_color_frame();
        self.wRGB,self.hRGB = self._args["kinectRGB"].color_frame_desc.Width,self._args["kinectRGB"].color_frame_desc.Height;
        self.RGBFrame = self.RGBFrame.reshape(self.hRGB,self.wRGB,-1).astype(np.uint8);
        self.RGBFrame = cv2.resize(self.RGBFrame, (int(self.wRGB/resize),int(self.hRGB/resize)));
        self.RGBFrame = cv2.cvtColor(self.RGBFrame, cv2.COLOR_BGR2RGB);
        self.RGBFrame = cv2.cvtColor(self.RGBFrame, cv2.COLOR_RGB2BGR);
        self.RGBFrame = cv2.flip(self.RGBFrame,0);
    
    def GetDepthFrame(self, resize) :
        
        self.DepthFrame = self._args["kinectDEPTH"].get_last_depth_frame();
        self.wDepth,self.hDepth = self._args["kinectDEPTH"].depth_frame_desc.Width,self._args["kinectDEPTH"].depth_frame_desc.Height;
        self.DepthFrame = self.DepthFrame.reshape(self.hDepth,self.wDepth,-1).astype(np.uint8);
        self.DepthFrame = cv2.resize(self.DepthFrame, (int(self.wDepth/resize),int(self.hDepth/resize)));
        self.DepthFrame = cv2.cvtColor(self.DepthFrame, cv2.COLOR_GRAY2BGR); 
        self.DepthFrame = cv2.flip(self.DepthFrame,0);
        self.DepthFrame = cv2.resize(self.DepthFrame, (self.wRGB,self.hRGB));

    def LoadRGBDepth(self, resizeRGB, resizeDepth) :
        
        self.GetRGBFrame(resizeRGB);
        self.GetDepthFrame(resizeDepth);
    
    def CreateDisplay(self, resizeRGB, resizeDepth) :
        
        self.LoadRGBDepth(resizeRGB, resizeDepth);
        self.hStack = np.hstack((self.RGBFrame, self.DepthFrame));
    
    def TestKinect(self, grid=False) : 
        
        while True :
            
            self.CreateDisplay(3,1);
            self.DepthFrameClone = self.DepthFrame.copy();
            
            if grid : 
                
                self.horizontalLength = 2*( tan( (self.kinectHorizontalAngle*(pi/180)) /2 )*self.kinectHeight );
                self.verticalLength = 2*( tan( (self.kinectVerticalAngle*(pi/180)) /2 )*self.kinectHeight );
                self.distanceRatio = int( (self.wRGB/self.horizontalLength) + (self.hRGB/self.verticalLength) /2 )
                
                self.horizontalPosition = int(( self.wRGB-(self._args["boxSize"]*self.distanceRatio) )/2);
                self.verticalPosition = int(( self.hRGB-(self._args["boxSize"]*self.distanceRatio) )/2);
                
                cv2.rectangle(self.DepthFrame, (self.horizontalPosition, self.verticalPosition),\
                              (int(self.horizontalPosition+self._args["boxSize"]*self.distanceRatio),\
                               int(self.verticalPosition+self._args["boxSize"]*self.distanceRatio)), (255,0,0), 1);     
                        
                for x in np.arange(self.horizontalPosition+self._args["gridRes"],\
                                   self.horizontalPosition+(self._args["boxSize"]*self.distanceRatio),self._args["gridRes"]) :
                    
                    for y in np.arange(self.verticalPosition+self._args["gridRes"],\
                                       self.verticalPosition+(self._args["boxSize"]*self.distanceRatio),self._args["gridRes"]) : 
                        
                        local_pixel_values = [];
                        
                        for x_pixel in np.arange(-self._args["gridRes"]/2,self._args["gridRes"]/2) :
                            
                            for y_pixel in np.arange(-self._args["gridRes"]/2,self._args["gridRes"]/2) :
                                
                                pixel_value = self.DepthFrameClone[int(y+y_pixel)][int(x+x_pixel)][0];
                                local_pixel_values.append(pixel_value);
                                
                        avg_local_value = sum(local_pixel_values)/len(local_pixel_values);
                        
                        if avg_local_value > self._args["depthMaxThresh"] : 
                            cv2.circle(self.DepthFrame,(x,y), self._args["gridRes"]-12, (255,0,0), 1);
                        elif avg_local_value < self._args["depthMinThresh"] : 
                            cv2.circle(self.DepthFrame,(x,y), self._args["gridRes"]-12, (0,0,255), 1);
                        else : 
                            cv2.circle(self.DepthFrame,(x,y), self._args["gridRes"]-12, (0,255,0), 1);
                        
                        cv2.putText(self.DepthFrame, str(int(avg_local_value)), (x-10,y+2),cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255));
            
            else :
                
                pass;
            
            cv2.imshow("RGBframe", self.RGBFrame); 
            cv2.imshow("DEPTHframe",self.DepthFrame); 
   
            if cv2.waitKey(1) & 0xFF == ord('q'):
                
                break;
                
        cv2.destroyAllWindows();
        
    def saveFrames(self) :
        
        self.tNow = time.time();
        self.LoadRGBDepth();
        self.frameCnt += 1;
        
        self.RGBWriter.write(self.RGBFrame);
        self.DepthWriter.write(self.DepthFrame);
        
    def PlayAndSave(self, display=True, samplingTime=60) :
        
        #Set timers
        #----------------------------------------------------------------------
        
        self.time = time.localtime(time.time());
        self.tStart = time.time();
        self.tNow = time.time();
        
        #Set counters
        #----------------------------------------------------------------------
        
        self.frameCnt = 0;
        
        #Set directories
        #----------------------------------------------------------------------
        
        self.dataDirName = '{0}-{1}-{2}_{3}-{4}-{5}'.format(self.time.tm_mday,\
                        self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                        self.time.tm_min,self.time.tm_sec);
                            
        self.dataDir = os.path.join(self._args["savingDir"],self.dataDirName);
        utils.CheckDirectoryExists(self.dataDir);
        
        if samplingTime != 1 :
        
            #Checking frame sizes
            #----------------------------------------------------------------------
            
            self.LoadRGBDepth(1,1);
            
            self.RGBString = self._args["rawVideoFileName"],self.time.tm_mday,\
                            self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                            self.time.tm_min,self.time.tm_sec;
        
            self.DepthString = self._args["depthVideoFileName8Bit"],self.time.tm_mday,\
                            self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                            self.time.tm_min,self.time.tm_sec;
                            
            self.TestRGBWriter = cv2.VideoWriter(os.path.join(self.dataDir,\
                                                'TestRGB.avi'),self._args["fourcc"],20,(self.wRGB,self.hRGB));
                                                    
            self.TestDepthWriter = cv2.VideoWriter(os.path.join(self.dataDir,\
                                    'TestDEPTH.avi'),self._args["fourcc"],20,(self.wDepth,self.hDepth));
                            
            #Launch framerate sampling
            #----------------------------------------------------------------------
            
            print("\n");
            print("[INFO] Starting framerate sampling for {0}s...".format(samplingTime));
            
            try :
                
                while self.tNow-self.tStart < samplingTime : 
                
                    self.saveFrames(self.TestRGBWriter,self.TestDepthWriter);
                        
                    if display :
                        
                        self.downSampledRGB = cv2.resize(self.RGBFrame, (0,0), fx=0.3, fy=0.3);
                        cv2.imshow('RGB',self.downSampledRGB);
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break;
                    
            except KeyboardInterrupt:
                
                pass;
        
            #Resets timers and counters
            #----------------------------------------------------------------------
            
            self.TestRGBWriter.release();
            self.TestDEPTHWriter.release();
            
            cv2.destroyAllWindows();
    
            self.sampledFrameRate = self.frameCnt/(self.tNow-self.tStart);
        
            os.remove(os.path.join(self.dataDir,'TestRGB.avi'));
            os.remove(os.path.join(self.dataDir,'TestDEPTH.avi'));
        
        else :
            
            self.LoadRGBDepth(1,1);
        
        self.RGBWriter = cv2.VideoWriter(os.path.join(self.dataDir,\
                                '{0}_{1}-{2}-{3}_{4}-{5}-{6}.avi'.format(*self.RGBString)),\
                                self._args["fourcc"],self.sampledFrameRate,(self.wRGB,self.hRGB));
                                                
        self.DepthWriter = cv2.VideoWriter(os.path.join(self.dataDir,\
                                '{0}_{1}-{2}-{3}_{4}-{5}-{6}.avi'.format(*self.DEPTH8BitString)),\
                                self._args["fourcc"],self.sampledFrameRate,(self.wDepth,self.hDepth));
        
        self.tStart = time.time();
        self.frameCnt = 0;  
        
        #Launch real saving
        #----------------------------------------------------------------------
        
        print("[INFO] Starting video recording...");
        
        try :
            
            while True :
                    
                self.saveFrames(self.RGBWriter,self.DepthWriter);
                
                if self.display :
                    
                    self.downSampledRGB = cv2.resize(self.RGBFrame, (0,0), fx=0.3, fy=0.3);
                    cv2.imshow('RGB',self.downSampledRGB);
                        
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break;
                    
        except KeyboardInterrupt:
                
            pass;
            
        #Stops stream saving and saves metadata
        #----------------------------------------------------------------------
        
        self.RGBWriter.release();
        self.DEPTH8BitWriter.release();
        
        self.tEnd = time.time();
        self.frameRate = self.frameCnt/(self.tEnd-self.tStart);
        
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
        sheet.write(2, 1, str(utils.HoursMinutesSeconds(self.tEnd-self.tStart)[0])+':'+\
                            str(utils.HoursMinutesSeconds(self.tEnd-self.tStart)[1])+':'+\
                            str(utils.HoursMinutesSeconds(self.tEnd-self.tStart)[2]));
        
        sheet.write(3, 0, "Sampled_Framerate");
        sheet.write(3, 1, self.sampledFrameRate);
        
        sheet.write(4, 0, "Real_Framerate");
        sheet.write(4, 1, self.frameRate);
        
        sheet.write(5, 0, "nFrames");
        sheet.write(5, 1, self.frameCnt);
        
        sheet.write(6, 0, "Fourcc");
        sheet.write(6, 1, self._args["fourcc"]);
        
        self.metaData.save(self.metaDataFile);