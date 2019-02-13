#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:05:37 2019

@author: tomtop
"""

import os;
import numpy as np;
import cv2;
import threading;
import xlwt;
import Queue;
import time;

import TopMouseTracker.Utilities as utils;

class MetaData() :
    
    def __init__(self,**kwargs) :
        
        self._args = kwargs;
        
        self.time = time.localtime(time.time());
        self.frameRate = 30.;
        
        self.dataDirName = '{0}-{1}-{2}_{3}-{4}-{5}'.format(self.time.tm_mday,\
                        self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                        self.time.tm_min,self.time.tm_sec);
                            
        self.dataDir = os.path.join(self._args["savingDir"],self.dataDirName);
        utils.CheckDirectoryExists(self.dataDir);
        
        #Checking frame sizes
        #----------------------------------------------------------------------
        
        testFrameRGB = self.GetFrame(self._args["kinectRGB"],"rgb",1);
        
        try :
            hRGB,wRGB,_ = testFrameRGB.shape;
        except : 
            hRGB,wRGB = testFrameRGB.shape;
        
        testFrameDEPTH,_ = self.GetFrame(self._args["kinectDEPTH"],"depth",1);
        
        try :
            hDEPTH,wDEPTH,_ = testFrameDEPTH.shape;
        except :
            hDEPTH,wDEPTH = testFrameDEPTH.shape;
        
        self.RGBString = self._args["rawVideoFileName"],self.time.tm_mday,\
                        self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                        self.time.tm_min,self.time.tm_sec;
    
        self.DEPTH8BitString = self._args["depthVideoFileName8Bit"],self.time.tm_mday,\
                        self.time.tm_mon,self.time.tm_year,self.time.tm_hour,\
                        self.time.tm_min,self.time.tm_sec;
                        
        self.RGBWriter = cv2.VideoWriter(os.path.join(self.dataDir,\
                                '{0}_{1}-{2}-{3}_{4}-{5}-{6}.avi'.format(*self.RGBString)),\
                                self._args["fourcc"],self.frameRate,(wRGB,hRGB));
                                                
        self.DEPTH8BitWriter = cv2.VideoWriter(os.path.join(self.dataDir,\
                                '{0}_{1}-{2}-{3}_{4}-{5}-{6}.avi'.format(*self.DEPTH8BitString)),\
                                self._args["fourcc"],self.frameRate,(wDEPTH,hDEPTH));
            
        def SaveMetaData(self) :
        
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
                        
            h,m,s = utils.HoursMinutesSeconds(self.tEnd-self.tStart);
            sheet.write(2, 0, "Elapsed_Time");
            sheet.write(2, 1, "{0}h-{1}m-{2}s".format(h,m,s));
            
            sheet.write(3, 0, "Sampled_Framerate");
            sheet.write(3, 1, self.sampledFrameRate);
            
            sheet.write(4, 0, "Real_Framerate");
            sheet.write(4, 1, self.frameRate);
            
            sheet.write(5, 0, "nFrames");
            sheet.write(5, 1, self.frameCnt);
            
            sheet.write(6, 0, "Fourcc");
            sheet.write(6, 1, self._args["fourcc"]);
            
            self.metaData.save(self.metaDataFile);

class ImageGrabber(threading.Thread):
    
    def __init__(self, camera, which, queue):
        
        threading.Thread.__init__(self);
        self._n = 0;
        self.camera, self.which, self.queue = camera, which, queue;

    def run(self):
        
        while True:
            
            if self.which == "rgb" :
                
                frame = self.camera.get_last_color_frame();
                self._W,self._H = self.camera.color_frame_desc.Width,self.camera.color_frame_desc.Height;
                frame = frame.reshape(self._H,self._W,-1).astype(np.uint8);
                frame = self.ConvertAndFlip(frame);
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); 
                #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR);
                frame = cv2.flip(frame,0);
                    
            if self.which == "depth" :
                
                frame = self.camera.get_last_depth_frame();
                self._W,self._H = self.camera.depth_frame_desc.Width,self.camera.depth_frame_desc.Height;
                frame = frame.reshape(self._H,self._W,-1).astype(np.uint16);
                frame = cv2.flip(frame,0);
                
            self.queue.put(frame);
            self._n += 1;
            
    def returnFrameNumber(self) :
        
        print("Number of frames : {0}".format(self._n));
        return self._n;      
            
class VideoWriter(threading.Thread):
    
    def __init__(self,writer,queue):
        
        threading.Thread.__init__(self);
        self.writer, self.queue = writer, queue;
        
        self.frameRate = 30.;
        self.dt = 1./self.frameRate;
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG');

    def run(self):
        
        self._n = 0; #number of frames
        self._nEverySecond = 0;
        self.frameRates = [];
        
        self.tStart = time.time(); #time when the writer started
        self.tBefore = time.time();
        self.tNow = time.time(); #current time
        self.tEverySecond = time.time();

        while True:
            
            if(not self.queue.empty()):
                
                self.tCurr = time.time();
                
                if self._n == 0 :
                    
                    self.frame = self.queue.get();
                    self.writer.write(self.frame);
                    self._n += 1;
                    self._nEverySecond += 1;
                    
                    self.tBefore = self.tCurr;
                    
                if self.tCurr-self.tBefore <= self.dt :
                    
                    time.sleep(self.dt-(self.tCurr-self.tBefore));
                    self.frame = self.queue.get();
                    self.writer.write(self.frame);
                    self._n += 1;
                    self._nEverySecond += 1;
                    
                    self.tBefore = self.tCurr;
                    
                else :
                    
                    print("Code is too slow!");
            
                if self.tCurr - self.tEverySecond > 1 :
                    
                    self.frameRates.append(self._nEverySecond/(self.tCurr-self.tEverySecond));
                    
                    self.tEverySecond = self.tCurr;
                    self.framesEverySecond = 0;
                    
                cv2.imshow("frame",self.frame);
                
                if cv2.waitKey(1) & 0XFF == ord("q") :
                    
                    break;
                
        self.writer.release();
    