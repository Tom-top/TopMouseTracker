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
    
    def __init__(self,limit=6,**kwargs) :
        
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
        self.distanceCorrected = [dist if dist > self._args["minDist"] and dist < self._args["maxDist"]else 0 for dist in self.distanceNormalized];
        self.distanceCumulative = list(np.cumsum(self.distanceCorrected));
        
        self.distanceNormalizedBefore = self.distanceNormalized[0:self._tStartBehav*self._args["framerate"]];
        self.areasBefore = self._areas[0:self._tStartBehav*self._args["framerate"]];
        
        if len(self.distanceNormalizedBefore) >= limit*3600*self._args["framerate"] :
            self.distanceNormalizedBefore = self.distanceNormalized[0:limit*3600*self._args["framerate"]];
            self.areasBefore = self._areas[0:limit*3600*self._args["framerate"]];
        
        self.distanceCorrectedBefore = [dist if dist > self._args["minDist"] and dist < self._args["maxDist"]else 0 for dist in self.distanceNormalizedBefore];
        self.distanceCumulativeBefore = list(np.cumsum(self.distanceCorrectedBefore));
            
        self.distanceNormalizedAfter = self.distanceNormalized[self._tStartBehav*self._args["framerate"]:self._Length*self._args["framerate"]];
        self.areasAfter = self._areas[self._tStartBehav*self._args["framerate"]:self._Length*self._args["framerate"]];
        
        if len(self.distanceNormalized)+len(self.distanceNormalizedBefore) >= limit*3600*self._args["framerate"] :
            self.distanceNormalizedAfter = self.distanceNormalized[self._tStartBehav*self._args["framerate"]:limit*3600*self._args["framerate"]];
            self.areasAfter = self._areas[self._tStartBehav*self._args["framerate"]:limit*3600*self._args["framerate"]];
        
        self.distanceCorrectedAfter = [dist if dist > self._args["minDist"] and dist < self._args["maxDist"]else 0 for dist in self.distanceNormalizedAfter];
        self.distanceCumulativeAfter = list(np.cumsum(self.distanceCorrectedAfter));
        self.distanceCumulativeAfter = [x+self.distanceCumulativeBefore[-1] for x in self.distanceCumulativeAfter];
        
    def CheckTracking(self) :
         
        fig = plt.figure();
        fig.suptitle("Mouse {0}".format(self._args["mouse"]), fontsize = 12, y = 1.01);
        
        ax0 = plt.subplot(311);
        ax0.plot(np.arange(0,len(self.distanceCorrectedBefore)),self.distanceCorrectedBefore,color='blue',alpha=0.5);
        ax0.plot(np.arange(len(self.distanceCorrectedBefore),len(self.distanceCorrectedBefore)+len(self.distanceCorrectedAfter)),self.distanceCorrectedAfter,color='red',alpha=0.5);
        ax0.set_title("Speed over time", fontsize = 10);
        
        ax1 = plt.subplot(312);
        ax1.plot(np.arange(0,len(self.distanceCumulativeBefore)),self.distanceCumulativeBefore,color='blue',alpha=0.5);
        ax1.plot(np.arange(len(self.distanceCumulativeBefore),len(self.distanceCumulativeBefore)+len(self.distanceCumulativeAfter)),self.distanceCumulativeAfter,color='red',alpha=0.5);
        ax1.set_title("Cumulative distance over time", fontsize = 10);
        
        ax2 = plt.subplot(313);
        ax2.plot(np.arange(0,len(self.areasBefore)),self.areasBefore,color='blue',alpha=0.5);
        ax2.plot(np.arange(len(self.areasBefore),len(self.areasBefore)+len(self.areasAfter)),self.areasAfter,color='red',alpha=0.5);
        ax2.set_title("Mask area over time", fontsize = 10);
        
        plt.tight_layout();
        
    def TrackingPlot(self,res,cBefore='b',cAfter='r',limit=6) :
        
        fig = plt.figure();
        ax0 = plt.subplot();

        ax0.set_xlim([0, int(self._ROIWidth)]);
        ax0.set_ylim([0, int(self._ROILength)]);
        
        self.distTraveledBeforeInitiation = 0;
        self.distTraveledAfterInitiation = 0;
        
        self.posBefore = self._positions[0:self._tStartBehav*self._args["framerate"]];
        
        if len(self.posBefore) >= limit*3600*self._args["framerate"] :
            self.posBefore = self._positions[0:limit*3600*self._args["framerate"]];
        
        self.posAfter = self._positions[self._tStartBehav*self._args["framerate"]:self._Length*self._args["framerate"]];
        
        if len(self.posBefore)+len(self.posAfter) >= limit*3600*self._args["framerate"] :
            self.posAfter = self._positions[self._tStartBehav*self._args["framerate"]:(limit*3600*self._args["framerate"])-len(self.posBefore)];
        
        self.filteredPosBefore = self.posBefore[0::res];
        self.filteredPosAfter = self.posAfter[0::res];
        
        self.befPlot = ax0.plot([x[0] for x in self.filteredPosBefore],[y[1] for y in self.filteredPosBefore],'-o',markersize=1,alpha=0.1,color='blue',label='Before Initiation');
        self.aftPlot = ax0.plot([x[0] for x in self.filteredPosAfter],[y[1] for y in self.filteredPosAfter],'-o',markersize=1,alpha=0.1,color='red',label='After Initiation');
        
        self.distTraveledBeforeInitiation = sum(self.distanceCorrected[0:self._tStartBehav*self._args["framerate"]]);
        
        if len(self.posBefore) >= limit*3600*self._args["framerate"] :
            self.distTraveledBeforeInitiation = sum(self.distanceCorrected[0:limit*3600*self._args["framerate"]]);
            
        self.distTraveledAfterInitiation = sum(self.distanceCorrected[self._tStartBehav*self._args["framerate"]:self._Length*self._args["framerate"]]);
            
        if len(self.posBefore)+len(self.posAfter) >= limit*3600*self._args["framerate"] :
            self.distTraveledAfterInitiation = sum(self.distanceCorrected[self._tStartBehav*self._args["framerate"]:(limit*3600*self._args["framerate"])-len(self.posBefore)]);
                    
        self.distTraveledBeforeInitiation = "%.2f" % (self.distTraveledBeforeInitiation/100);
        self.distTraveledAfterInitiation = "%.2f" % (self.distTraveledAfterInitiation/100);
        
        hB,mB,sB = utils.HoursMinutesSeconds(len(self.posBefore)/self._args["framerate"]);
        hA,mA,sA = utils.HoursMinutesSeconds(len(self.posAfter)/self._args["framerate"]);
        
        fontBefore = { "size" : 10,
                      "color" : cBefore,
                      "alpha" : 0.5,
        };
                      
        fontAfter = { "size" : 10,
                      "color" : cAfter,
                      "alpha" : 0.5,
        };
        
        ax0.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
        ax0.set_title("Tracking Mouse {0}".format(self._mouse),position=(0.83, 1.025));
        
        ax0.text(int(self._ROIWidth)/3, -27, 'Time before : {0}h {1}m {2}s'.format(str(hB),str(mB),str(sB)),fontdict=fontBefore);
        ax0.text(int(self._ROIWidth)/3, -19, 'Dist before : {0}'.format(self.distTraveledBeforeInitiation)+'m',fontdict=fontBefore);
        ax0.text(int(self._ROIWidth)/3, -11, 'Time after : {0}h {1}m {2}s'.format(str(hA),str(mA),str(sA)),fontdict=fontAfter);
        ax0.text(int(self._ROIWidth)/3, -3, 'Dist after : {0}'.format(self.distTraveledAfterInitiation)+'m',fontdict=fontAfter)
        ax0.legend(handles = [self.befPlot[0],self.aftPlot[0]],loc=(0,1.05),shadow=True);
        
        plt.gca().invert_yaxis();
        
        plt.tight_layout();
        
    def CompleteTrackingPlot(self,res,cBefore='b',cAfter='r',limit=6,save=False) :
        
        fig = plt.figure(figsize=(20,10));
        fig.suptitle("Tracking Mouse {0}".format(self._args["mouse"]), fontsize = 12, y = 0.97);

        ax0 = plt.subplot2grid((3, 4), (0, 3));
        #ax0 = plt.subplot(3,4,4);
        ax0.plot(np.arange(0,len(self.distanceCorrectedBefore)),self.distanceCorrectedBefore,color='blue',alpha=0.5);
        ax0.plot(np.arange(len(self.distanceCorrectedBefore),len(self.distanceCorrectedBefore)+len(self.distanceCorrectedAfter)),self.distanceCorrectedAfter,color='red',alpha=0.5);
        ax0.set_title("Speed over time (cm/s)", fontsize = 10);
        ax0.set_ylabel("Speed (cm/s)");
        ax0.tick_params(bottom=False,labelbottom=False);
        
        
        ax1 = plt.subplot2grid((3, 4), (1, 3));
        #ax1 = plt.subplot(3,4,8);
        ax1.plot(np.arange(0,len(self.distanceCumulativeBefore)),self.distanceCumulativeBefore,color='blue',alpha=0.5);
        ax1.plot(np.arange(len(self.distanceCumulativeBefore),len(self.distanceCumulativeBefore)+len(self.distanceCumulativeAfter)),self.distanceCumulativeAfter,color='red',alpha=0.5);
        ax1.set_title("Cumulative distance over time", fontsize = 10);
        ax1.set_ylabel("Cumulative distance (cm)");
        ax1.tick_params(bottom=False,labelbottom=False);
        
        ax2 = plt.subplot2grid((3, 4), (2, 3));
        #ax2 = plt.subplot(3,4,12);
        ax2.plot(np.arange(0,len(self.areasBefore)),self.areasBefore,color='blue',alpha=0.5);
        ax2.plot(np.arange(len(self.areasBefore),len(self.areasBefore)+len(self.areasAfter)),self.areasAfter,color='red',alpha=0.5);
        ax2.set_title("Mask area over time", fontsize = 10);
        ax2.set_ylabel("Mask area (px^2)");
        ax2.set_xlabel("time (h)");
        ax2.set_xticks(np.arange(0,limit*3600*self._args["framerate"],100000));
        ax2.set_xticklabels(np.arange(0,limit+1,1));
        
        ax3 = plt.subplot2grid((3, 4), (0, 0), rowspan=3, colspan=3);
        
        ax3.set_xlim([0, int(self._ROIWidth)]);
        ax3.set_ylim([0, int(self._ROILength)]);
        
        self.distTraveledBeforeInitiation = 0;
        self.distTraveledAfterInitiation = 0;
        
        self.posBefore = self._positions[0:self._tStartBehav*self._args["framerate"]];
        
        if len(self.posBefore) >= limit*3600*self._args["framerate"] :
            self.posBefore = self._positions[0:limit*3600*self._args["framerate"]];
        
        self.posAfter = self._positions[self._tStartBehav*self._args["framerate"]:self._Length*self._args["framerate"]];
        
        if len(self.posBefore)+len(self.posAfter) >= limit*3600*self._args["framerate"] :
            self.posAfter = self._positions[self._tStartBehav*self._args["framerate"]:limit*3600*self._args["framerate"]];
        
        self.filteredPosBefore = self.posBefore[0::res];
        self.filteredPosAfter = self.posAfter[0::res];
        
        self.befPlot = ax3.plot([x[0] for x in self.filteredPosBefore],[y[1] for y in self.filteredPosBefore],'-',markersize=1,alpha=0.5,color='blue',label='Before Initiation');
        self.aftPlot = ax3.plot([x[0] for x in self.filteredPosAfter],[y[1] for y in self.filteredPosAfter],'-',markersize=1,alpha=0.5,color='red',label='After Initiation');
        
        self.distTraveledBeforeInitiation = sum(self.distanceCorrected[0:self._tStartBehav*self._args["framerate"]]);
        
        if len(self.posBefore) >= limit*3600*self._args["framerate"] :
            self.distTraveledBeforeInitiation = sum(self.distanceCorrected[0:limit*3600*self._args["framerate"]]);
            
        self.distTraveledAfterInitiation = sum(self.distanceCorrected[self._tStartBehav*self._args["framerate"]:self._Length*self._args["framerate"]]);
            
        if len(self.posBefore)+len(self.posAfter) >= limit*3600*self._args["framerate"] :
            self.distTraveledAfterInitiation = sum(self.distanceCorrected[self._tStartBehav*self._args["framerate"]:limit*3600*self._args["framerate"]]);
                    
        self.distTraveledBeforeInitiation = "%.2f" % (self.distTraveledBeforeInitiation/100);
        self.distTraveledAfterInitiation = "%.2f" % (self.distTraveledAfterInitiation/100);
        
        hB,mB,sB = utils.HoursMinutesSeconds(len(self.posBefore)/self._args["framerate"]);
        hA,mA,sA = utils.HoursMinutesSeconds(len(self.posAfter)/self._args["framerate"]);
        
        fontBefore = { "size" : 10,
                      "color" : cBefore,
                      "alpha" : 0.5,
        };
                      
        fontAfter = { "size" : 10,
                      "color" : cAfter,
                      "alpha" : 0.5,
        };
        
        ax3.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
        
        ax3.text(int(self._ROIWidth)/6, -7, 'Time before : {0}h {1}m {2}s'.format(str(hB),str(mB),str(sB)),fontdict=fontBefore);
        ax3.text(int(self._ROIWidth)/6, -3, 'Dist before : {0}'.format(self.distTraveledBeforeInitiation)+'m',fontdict=fontBefore);
        ax3.text(int(self._ROIWidth)/6+40, -7, 'Time after : {0}h {1}m {2}s'.format(str(hA),str(mA),str(sA)),fontdict=fontAfter);
        ax3.text(int(self._ROIWidth)/6+40, -3, 'Dist after : {0}'.format(self.distTraveledAfterInitiation)+'m',fontdict=fontAfter)
        ax3.legend(handles = [self.befPlot[0],self.aftPlot[0]],loc=(0,1.02),shadow=True);
        
        plt.gca().invert_yaxis();
        
        plt.tight_layout();
        
        if save :
            
            plt.savefig(os.path.join(self._args["baseDir"],"Complete_Tracking_Mouse_{0}".format(self._mouse)))
        
    def HeatMapPlot(self,gridsize) :
        
        fig = plt.figure();
        ax0 = plt.subplot();

        ax0.set_xlim([0, int(self._ROIWidth)]);
        ax0.set_ylim([0, int(self._ROILength)]);
        
        self._x = [x[0] for x in self._positions]+[0,int(self._ROIWidth)];
        self._y = [x[1] for x in self._positions]+[int(self._ROILength),0];
        
        cmap = plt.get_cmap('jet');
        norm = mcolors.LogNorm();
        
        ax0.hexbin(self._x, self._y, gridsize=gridsize, cmap=cmap, norm=norm);
        ax0.set_title('Tracking Mouse {0}'.format(self._mouse));
        ax0.tick_params(top=False, bottom=False, left=False, right=False);
        
        plt.gca().invert_yaxis();
        
        