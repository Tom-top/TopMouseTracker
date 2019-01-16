#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:06:08 2019

@author: tomtop
"""

import os;
import pandas as pd;
import numpy as np;
from math import sqrt;
import matplotlib.pyplot as plt;
import matplotlib.colors as mcolors;

import TopMouseTracker.Utilities as utils;

class Plot() :
    
    def __init__(self,**kwargs) :
        
        self._args = kwargs;
        self._mouse = self._args["mouse"]; #Name of the mouse
        
        videoInfoWorkbook = pd.read_excel(self._args["baseDir"]+'/Mice_Video_Info.xlsx'); #Load video info excel sheet
        videoInfo = videoInfoWorkbook.as_matrix(); #Transforms it into a matrix
        
        for metaData in videoInfo :
            if str(metaData[0]) == self._mouse :
                self._tStart = int(metaData[1]); #Moment at which the cotton is added (s)
                self._tStartBehav = int(metaData[2]); #Moment at which the mouse start nest-building (s)
                self._tEnd = [int(metaData[3]),int(metaData[4]),\
                             int(metaData[5]),int(metaData[6])]; #Length of each video of the experiment (max 4 videos)
                self._Length = sum(self._tEnd);
  
        self._positions = np.load(os.path.join(self._args["directory"],'Mouse_Data_All_'+self._args["mouse"]+'_Points.npy'));
        self._refPt = np.load(os.path.join(self._args["directory"],'Mouse_Data_All_'+self._args["mouse"]+'_refPt.npy'));
        self._areas = np.load(os.path.join(self._args["directory"],'Mouse_Data_All_'+self._args["mouse"]+'_Areas.npy'));
        
        self._cageWidth = self._args["cageWidth"];
        self._cageLength = self._args["cageLength"];
        
        self._ROIWidth = abs(self._refPt[0][0]-self._refPt[1][0]);
        self._ROILength = abs(self._refPt[0][1]-self._refPt[1][1]);
        
        self._averageRatio = ( (self._ROIWidth/self._cageWidth) + (self._ROILength/self._cageLength) )/2;
        
        self.distance = [sqrt((self._positions[n+1][0]-self._positions[n][0])**2+\
                          (self._positions[n+1][1]-self._positions[n][1])**2) for n in range(len(self._positions)) if n+1 < len(self._positions)];
                
        self.distanceNormalized = [dist/self._averageRatio for dist in self.distance];
        self.distanceCorrected = [dist if dist > self._args["minDist"] else 0 for dist in self.distanceNormalized];
        self.distanceCumulative = list(np.cumsum(self.distanceCorrected));
        
    def CheckTracking(self) :
         
        fig = plt.figure();
        fig.suptitle("Mouse {0}".format(self._args["mouse"]), fontsize = 12, y = 1.01);
        
        ax0 = plt.subplot(311);
        ax0.plot(self.distanceCorrected);
        ax0.set_title("Speed over time", fontsize = 10);
        
        ax1 = plt.subplot(312);
        ax1.plot(self.distanceCumulative);
        ax1.set_title("Cumulative distance over time", fontsize = 10);
        
        ax2 = plt.subplot(313);
        ax2.plot(self._areas);
        ax2.set_title("Mask area over time", fontsize = 10);
        
        plt.tight_layout();
        
    def TrackingPlot(self,res,cBefore='b',cAfter='r') :
        
        fig = plt.figure();
        ax0 = plt.subplot();

        ax0.set_xlim([0, int(self._ROIWidth)]);
        ax0.set_ylim([0, int(self._ROILength)]);
        
        self.distTraveledBeforeInitiation = 0;
        self.distTraveledAfterInitiation = 0;
        
        self.posBefore = self._positions[self._tStart*self._args["framerate"]:self._tStartBehav*self._args["framerate"]];
        self.posAfter = self._positions[self._tStartBehav*self._args["framerate"]:self._Length*self._args["framerate"]];
        
        self.filteredPosBefore = self.posBefore[0::res];
        self.filteredPosAfter = self.posAfter[0::res];
        
        ax0.plot([x[0] for x in self.filteredPosBefore],[y[1] for y in self.filteredPosBefore],'-o',markersize=1,alpha=0.1,color='blue');
        ax0.plot([x[0] for x in self.filteredPosAfter],[y[1] for y in self.filteredPosAfter],'-o',markersize=1,alpha=0.1,color='red');
        
        for i in range(len(self._positions)-1) :
            
            #if i%res == 0 :
                
                #self.percentage = (float(i)/float(len(self._positions)-1))*100;
                #utils.PrintColoredMessage("{0} percent of points ploted...".format(self.percentage),"darkgreen");
                
            dist = self.distanceNormalized[i];
            
            #----------------------------------------------------------------------------------------------------------------------------------
            #Creates line plot from mouse position over time and computes distance traveled
            
            if i > self._tStart*self._args["framerate"] and i <= self._tStartBehav*self._args["framerate"] :
                                     
                if int(dist) > self._args["minDist"] and int(dist) < self._args["maxDist"] :
                    
                    self.distTraveledBeforeInitiation += dist;
            
            elif i > self._tStartBehav*self._args["framerate"] and i <= self._Length*self._args["framerate"] :
                                    
                if int(dist) > self._args["minDist"] and int(dist) < self._args["maxDist"] :
                    
                    self.distTraveledAfterInitiation += dist;
                    
        self.distTraveledBeforeInitiation = "%.2f" % self.distTraveledBeforeInitiation;
        self.distTraveledAfterInitiation = "%.2f" % self.distTraveledAfterInitiation;
        
        self.timeBeforeInitiation = self._tStartBehav - self._tStart;
        self.timeAfterInitiation = self._Length - self._tStartBehav;
        
        #hB,mB,sB = utils.hoursMinutesSeconds(self.timeBeforeInitiation);
        #hA,mA,sA = utils.hoursMinutesSeconds(self.timeAfterInitiation);
        
        fontBefore = { "size" : 10,
                      "color" : cBefore,
                      "alpha" : 0.5,
        };
                      
        fontAfter = { "size" : 10,
                      "color" : cAfter,
                      "alpha" : 0.5,
        };
        
        ax0.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
        ax0.set_title("Tracking Mouse "+self._mouse,position=(0.87, 1.025));
        #ax0.text(int(self._ROIWidth)/2-8, int(self._ROILength)+(int(self._ROILength)*0.07), 'Time before : {0}h {1}m {2}s'.format(str(hB),str(mB),str(sB)),fontdict=fontBefore);
        #ax0.text(int(self._ROIWidth)/2-8, int(self._ROILength)+(int(self._ROILength)*0.01), 'Dist before : {0}'.format(self.distTraveledBeforeInitiation)+'m',fontdict=fontBefore);
        #ax0.text(int(self._ROIWidth)/2-8, int(self._ROILength)+(int(self._ROILength)*0.10), 'Time after : {0}h {1}m {2}s'.format(str(hA),str(mA),str(sA)),fontdict=fontAfter);
        #ax0.text(int(self._ROIWidth)/2-8, int(self._ROILength)+(int(self._ROILength)*0.04), 'Dist after : {0}'.format(self.distTraveledAfterInitiation)+'m',fontdict=fontAfter)
        #ax0.legend(handles = [redPlot[0],bluePlot[0]],loc=2,bbox_to_anchor=(-0.01,1.13),shadow=True);
        
    def HeatMapPlot(self,gridsize) :
        
        fig = plt.figure();
        ax0 = plt.subplot();

        ax0.set_xlim([0, int(self._ROIWidth)]);
        ax0.set_ylim([0, int(self._ROILength)]);
        
        self._x = [x[0] for x in self._positions];
        self._y = [x[1] for x in self._positions];
        
        cmap = plt.get_cmap('jet');
        norm = mcolors.LogNorm();
        
        ax0.hexbin(self._x, self._y, gridsize=gridsize, cmap=cmap, norm=norm);
        ax0.set_title('Heatmap Mouse {0}'.format(self._mouse));
        
        