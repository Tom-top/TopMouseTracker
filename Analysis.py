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
import matplotlib.patches as patches;
import peakutils;
from matplotlib.widgets import MultiCursor;
from scipy import optimize;

import TopMouseTracker.Utilities as utils;
import TopMouseTracker._Tracker as tracker;
from scipy.ndimage.filters import gaussian_filter;
import matplotlib.cm as cm;

class Plot(tracker.TopMouseTracker) :
    
    def __init__(self,**kwargs) :
        
        tracker.TopMouseTracker.__init__(self,**kwargs);
  
        self._positions = np.load(os.path.join(self._args["main"]["resultDir"],'Data_'+self._args["main"]["mouse"]+'_Points.npy'));
        self._refPt = np.load(os.path.join(self._args["main"]["resultDir"],'Data_'+self._args["main"]["mouse"]+'_refPt.npy'));
        self._areas = np.load(os.path.join(self._args["main"]["resultDir"],'Data_'+self._args["main"]["mouse"]+'_Areas.npy'));
        self._cottonAveragePixelIntensities = np.load(os.path.join(self._args["main"]["resultDir"],'Data_'+self._args["main"]["mouse"]+'_CottonPixelIntensities.npy'));
        self._cottonSpread = np.load(os.path.join(self._args["main"]["resultDir"],'Data_'+self._args["main"]["mouse"]+'_CottonSpread.npy'));
        
        self.upLeftX = int(self._refPt[0][0]); #Defines the Up Left ROI corner X coordinates
        self.upLeftY = int(self._refPt[0][1]); #Defines the Up Left ROI corner Y coordinates
        self.lowRightX = int(self._refPt[1][0]); #Defines the Low Right ROI corner X coordinates
        self.lowRightY = int(self._refPt[1][1]); #Defines the Low Right ROI corner Y coordinates
        
        self.distanceRatio = (abs(self.upLeftX-self.lowRightX)/self._args["segmentation"]["cageLength"]+\
                              abs(self.upLeftY-self.lowRightY)/self._args["segmentation"]["cageWidth"])/2; #Defines the resizing factor for the cage
                              
        self.ROIWidth = abs(self.lowRightX-self.upLeftX);
        self.ROILength = abs(self.lowRightY-self.upLeftY);
        
        self.distance = [sqrt((self._positions[n+1][0]-self._positions[n][0])**2+\
                          (self._positions[n+1][1]-self._positions[n][1])**2) for n in range(len(self._positions)) if n+1 < len(self._positions)];
                
        self.distanceNormalized = [dist/self.distanceRatio for dist in self.distance];
        self.distanceCorrected = [dist if dist > self._args["plot"]["minDist"] and dist < self._args["plot"]["maxDist"]else 0 for dist in self.distanceNormalized];
        self.distanceCumulative = list(np.cumsum(self.distanceCorrected));
        
        self.distanceNormalizedBefore = self.distanceNormalized[0:int(self._tStartBehav*np.mean(self._framerate))];
        self.areasBefore = self._areas[0:int(self._tStartBehav*np.mean(self._framerate))];
        self.cottonBefore = self._cottonAveragePixelIntensities[0:int(self._tStartBehav*np.mean(self._framerate))];
        self.spreadBefore = self._cottonSpread[0:int(self._tStartBehav*np.mean(self._framerate))];
        
        if len(self.distanceNormalizedBefore) >= self._args["plot"]["limit"]*3600*np.mean(self._framerate) :
            self.distanceNormalizedBefore = self.distanceNormalized[0:int(self._args["plot"]["limit"]*3600*np.mean(self._framerate))];
            self.areasBefore = self._areas[0:int(self._args["plot"]["limit"]*3600*np.mean(self._framerate))];
            self.cottonBefore = self._cottonAveragePixelIntensities[0:int(self._args["plot"]["limit"]*3600*np.mean(self._framerate))];
            self.spreadBefore = self._cottonSpread[0:int(self._args["plot"]["limit"]*3600*np.mean(self._framerate))];
        
        self.distanceCorrectedBefore = [dist if dist > self._args["plot"]["minDist"] and dist < self._args["plot"]["maxDist"]else 0 for dist in self.distanceNormalizedBefore];
        self.distanceCumulativeBefore = list(np.cumsum(self.distanceCorrectedBefore));
        
        #######################################
        self._Length = sum(self._tEnd)
        
        self.distanceNormalizedAfter = self.distanceNormalized[int(self._tStartBehav*np.mean(self._framerate)):int(self._Length*np.mean(self._framerate))];
        self.areasAfter = self._areas[int(self._tStartBehav*np.mean(self._framerate)):int(self._Length*np.mean(self._framerate))];
        self.cottonAfter = self._cottonAveragePixelIntensities[int(self._tStartBehav*np.mean(self._framerate)):int(self._Length*np.mean(self._framerate))];
        self.spreadAfter = self._cottonSpread[int(self._tStartBehav*np.mean(self._framerate)):int(self._Length*np.mean(self._framerate))];
        
        if len(self.distanceNormalized)+len(self.distanceNormalizedBefore) >= self._args["plot"]["limit"]*3600*np.mean(self._framerate) :
            self.distanceNormalizedAfter = self.distanceNormalized[int(self._tStartBehav*np.mean(self._framerate)):int(self._args["plot"]["limit"]*3600*np.mean(self._framerate))];
            self.areasAfter = self._areas[int(self._tStartBehav*np.mean(self._framerate)):int(self._args["plot"]["limit"]*3600*np.mean(self._framerate))];
            self.cottonAfter = self._cottonAveragePixelIntensities[int(self._tStartBehav*np.mean(self._framerate)):int(self._args["plot"]["limit"]*3600*np.mean(self._framerate))];
            self.spreadAfter = self._cottonSpread[int(self._tStartBehav*np.mean(self._framerate)):int(self._args["plot"]["limit"]*3600*np.mean(self._framerate))];
        
        self.distanceCorrectedAfter = [dist if dist > self._args["plot"]["minDist"] and dist < self._args["plot"]["maxDist"]else 0 for dist in self.distanceNormalizedAfter];
        self.distanceCumulativeAfter = list(np.cumsum(self.distanceCorrectedAfter));
        
        try :
            self.distanceCumulativeAfter = [x+self.distanceCumulativeBefore[-1] for x in self.distanceCumulativeAfter];
        except :
            self.distanceCumulativeAfter = [x for x in self.distanceCumulativeAfter];
        
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
        
    def TrackingPlot(self,res,plotB=True,plotA=True,cBefore='b',cAfter='r',limit=6) :
        
        fig = plt.figure(figsize=(12,7));
        ax0 = plt.subplot();

        ax0.set_xlim([0, int(self.ROIWidth)]);
        ax0.set_ylim([0, int(self.ROILength)]);
        
        self.distTraveledBeforeInitiation = 0;
        self.distTraveledAfterInitiation = 0;
        
        self._framerate = int(self._framerate);
        
        self.posBefore = self._positions[0:self._tStartBehav*self._framerate];
        
        if len(self.posBefore) >= limit*3600*self._framerate :
            self.posBefore = self._positions[0:limit*3600*self._framerate];
        
        self.posAfter = self._positions[self._tStartBehav*self._framerate:self._Length*self._framerate];
        
        if len(self.posBefore)+len(self.posAfter) >= limit*3600*self._framerate :
            self.posAfter = self._positions[self._tStartBehav*self._framerate:(limit*3600*self._framerate)-len(self.posBefore)];
        
        self.filteredPosBefore = self.posBefore[0::res];
        self.filteredPosAfter = self.posAfter[0::res];
        
        if plotB :
            self.befPlot = ax0.plot([x[0] for x in self.filteredPosBefore],[y[1] for y in self.filteredPosBefore],'-o',markersize=1,alpha=0.1,solid_capstyle="butt",color=cBefore,label='Before Initiation');
        if plotA :
            self.aftPlot = ax0.plot([x[0] for x in self.filteredPosAfter],[y[1] for y in self.filteredPosAfter],'-o',markersize=1,alpha=0.1,solid_capstyle="butt",color=cAfter,label='After Initiation');
        
        self.distTraveledBeforeInitiation = sum(self.distanceCorrected[0:self._tStartBehav*self._framerate]);
        
        if len(self.posBefore) >= limit*3600*self._framerate :
            self.distTraveledBeforeInitiation = sum(self.distanceCorrected[0:limit*3600*self._framerate]);
            
        self.distTraveledAfterInitiation = sum(self.distanceCorrected[self._tStartBehav*self._framerate:self._Length*self._framerate]);
            
        if len(self.posBefore)+len(self.posAfter) >= limit*3600*self._framerate :
            self.distTraveledAfterInitiation = sum(self.distanceCorrected[self._tStartBehav*self._framerate:(limit*3600*self._framerate)-len(self.posBefore)]);
                    
        self.distTraveledBeforeInitiation = "%.2f" % (self.distTraveledBeforeInitiation/100);
        self.distTraveledAfterInitiation = "%.2f" % (self.distTraveledAfterInitiation/100);
        
        hB,mB,sB = utils.HoursMinutesSeconds(len(self.posBefore)/self._framerate);
        hA,mA,sA = utils.HoursMinutesSeconds(len(self.posAfter)/self._framerate);
        
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
        
        ax0.text(int(self.ROIWidth)/3, -27, 'Time before : {0}h {1}m {2}s'.format(str(hB),str(mB),str(sB)),fontdict=fontBefore);
        ax0.text(int(self.ROIWidth)/3, -19, 'Dist before : {0}'.format(self.distTraveledBeforeInitiation)+'m',fontdict=fontBefore);
        ax0.text(int(self.ROIWidth)/3, -11, 'Time after : {0}h {1}m {2}s'.format(str(hA),str(mA),str(sA)),fontdict=fontAfter);
        ax0.text(int(self.ROIWidth)/3, -3, 'Dist after : {0}'.format(self.distTraveledAfterInitiation)+'m',fontdict=fontAfter)
        
        if plotB and plotA :
            ax0.legend(handles = [self.befPlot[0],self.aftPlot[0]],loc=(0,1.05),shadow=True);
        if plotB and not plotA : 
            ax0.legend(handles = [self.befPlot[0]],loc=(0,1.05),shadow=True);
        if not plotB and plotA :
            ax0.legend(handles = [self.aftPlot[0]],loc=(0,1.05),shadow=True);
        
        plt.gca().invert_yaxis();
        
        plt.tight_layout();
        
        if self._args["plot"]["save"] :
            
            if plotB and plotA :
            
                plt.savefig(os.path.join(self._args["main"]["resultDir"],"Tracking_Mouse_{0}".format(self._mouse)));
                
            if plotB and not plotA : 
                
                plt.savefig(os.path.join(self._args["main"]["resultDir"],"Tracking_Before_Mouse_{0}".format(self._mouse)));
                
            if not plotB and plotA :
                
                plt.savefig(os.path.join(self._args["main"]["resultDir"],"Tracking_After_Mouse_{0}".format(self._mouse)));
        
    def CompleteTrackingPlot(self,cBefore='b',cAfter='r',alpha=0.1, line=True, res=5, rasterSpread=10) :
        
        self.res = np.mean(self._framerate)*res
        
        fig = plt.figure(figsize=(20,10));
        fig.suptitle("Tracking Mouse {0}".format(self._args["main"]["mouse"]), fontsize = 12, y = 0.97);


        ax0 = plt.subplot2grid((5, 4), (0, 3));
        #ax0 = plt.subplot(3,4,4);
        #ax0.plot(np.arange(0,len(self.distanceCorrectedBefore)),self.distanceCorrectedBefore,color='blue',alpha=0.5);
        Before = [np.mean(self.distanceCorrectedBefore[int(i):int(i+self.res)]) for i in np.arange(0,len(self.distanceCorrectedBefore),self.res)];
        ax0.plot(np.arange(0,len(Before)),Before,color=cBefore,alpha=0.5);
        #ax0.plot(np.arange(len(self.distanceCorrectedBefore),len(self.distanceCorrectedBefore)+len(self.distanceCorrectedAfter)),self.distanceCorrectedAfter,color='red',alpha=0.5);
        After = [np.mean(self.distanceCorrectedAfter[int(i):int(i+self.res)]) for i in np.arange(0,len(self.distanceCorrectedAfter),self.res)];
        ax0.plot(np.arange(len(Before),len(Before)+len(After)),After,color=cAfter,alpha=0.5);
        ax0.set_title("Speed over time (cm/s)", fontsize = 10);
        ax0.set_ylabel("Speed (cm/s)");
        ax0.set_xticks(np.arange(0,(self._args["plot"]["limit"]*3600)/res,3600/res));
        ax0.tick_params(bottom=False,labelbottom=False);
        ax0.set_xlim([0,len(Before+After)])
        
        
        ax1 = plt.subplot2grid((5, 4), (1, 3));
        #ax1 = plt.subplot(3,4,8);
        #ax1.plot(np.arange(0,len(self.distanceCumulativeBefore)),self.distanceCumulativeBefore,color='blue',alpha=0.5);
        Before = [np.mean(self.distanceCumulativeBefore[int(i):int(i+self.res)]) for i in np.arange(0,len(self.distanceCumulativeBefore),self.res)]
        ax1.plot(np.arange(0,len(Before)),Before,color=cBefore,alpha=0.5);
        #ax1.plot(np.arange(len(self.distanceCumulativeBefore),len(self.distanceCumulativeBefore)+len(self.distanceCumulativeAfter)),self.distanceCumulativeAfter,color='red',alpha=0.5);
        After = [np.mean(self.distanceCumulativeAfter[int(i):int(i+self.res)]) for i in np.arange(0,len(self.distanceCumulativeAfter),self.res)];
        ax1.plot(np.arange(len(Before),len(Before)+len(After)),After,color=cAfter,alpha=0.5);
        ax1.set_title("Cumulative distance over time", fontsize = 10);
        ax1.set_ylabel("Cumulative distance (cm)");
        ax1.set_xticks(np.arange(0,(self._args["plot"]["limit"]*3600)/res,3600/res));
        ax1.tick_params(bottom=False,labelbottom=False);
        ax1.set_xlim([0,len(Before+After)])
        
        
#        ax2 = plt.subplot2grid((5, 4), (2, 3));
#        #ax2 = plt.subplot(3,4,12);
#        #ax2.plot(np.arange(0,len(self.areasBefore)),self.areasBefore,color='blue',alpha=0.5);
#        Before = [np.mean(self.areasBefore[int(i):int(i+self.res)]) for i in np.arange(0,len(self.areasBefore),self.res)];
#        ax2.plot(np.arange(0,len(Before)),Before,color=cBefore,alpha=0.5);
#        #ax2.plot(np.arange(len(self.areasBefore),len(self.areasBefore)+len(self.areasAfter)),self.areasAfter,color='red',alpha=0.5);
#        After = [np.mean(self.areasAfter[int(i):int(i+self.res)]) for i in np.arange(0,len(self.areasAfter),self.res)];
#        ax2.plot(np.arange(len(Before),len(Before)+len(After)),After,color=cAfter,alpha=0.5);
#        ax2.set_title("Mask area over time", fontsize = 10);
#        ax2.set_ylabel("Mask area (px^2)");
#        ax2.set_xticks(np.arange(0,(self._args["plot"]["limit"]*3600)/res,3600/res));
#        ax2.tick_params(bottom=False,labelbottom=False);
#        ax2.set_xlim([0,len(Before+After)])
        
        
        ax2 = plt.subplot2grid((5, 4), (2, 3));
        #ax3.plot(np.arange(0,len(self.cottonBefore)),self.cottonBefore,color='blue',alpha=0.5);
        Before = [np.mean(self.cottonBefore[int(i):int(i+self.res)]) for i in np.arange(0,len(self.cottonBefore),self.res)];
        ax2.plot(np.arange(0,len(Before)),Before,color=cBefore,alpha=0.5);
        #ax3.plot(np.arange(len(self.cottonBefore),len(self.cottonBefore)+len(self.cottonAfter)),self.cottonAfter,color='red',alpha=0.5);
        After = [np.mean(self.cottonAfter[int(i):int(i+self.res)]) for i in np.arange(0,len(self.cottonAfter),self.res)];
        ax2.plot(np.arange(len(Before),len(Before)+len(After)),After,color=cAfter,alpha=0.5);
        
        All = Before+After;
        
        peaks = [];
        
        for i,data in enumerate(All) :

            if i <= len(All)-2 :
            
                if All[i]+1 <= All[i+1] or All[i]-1 >= All[i+1] :
                    
                    peaks.append(i);
                
        
        #peaks = peakutils.indexes(All, thres=0.8, min_dist=1); 
        
        #print(peaks)
        #ax3.scatter(peaks,[All[i] for i in peaks], c='black');
        
        ax2.set_title("Cotton height over time", fontsize = 10);
        ax2.set_ylabel("Average pixel intensity *8bit (a.u)");
        ax2.set_xticks(np.arange(0,(self._args["plot"]["limit"]*3600)/res,3600/res));
        ax2.tick_params(bottom=False,labelbottom=False);
        ax2.set_xlim([0,len(Before+After)]);
        
        ax3 = plt.subplot2grid((5, 4), (3, 3));
        
        if rasterSpread == None :
            if All != [] : 
                rasterSpread = (self._args["plot"]["limit"]+1)/(len(All));
        
        for peak in peaks :
            ax3.add_patch(patches.Rectangle((peak, 0), rasterSpread, 1,color=cAfter,alpha=0.1));
        
        ax3.set_title("Nest-building activity over time", fontsize = 10);
        ax3.set_ylabel("Nest-building activity");
        ax3.set_xlabel("time (h)");
        ax3.set_xticks(np.arange(0,(self._args["plot"]["limit"]*3600)/res,3600/res));
        ax3.set_xticklabels(np.arange(0,self._args["plot"]["limit"]+1,1));
        #print(len(All))
        ax3.set_xlim([0,len(Before+After)])
        
        ax4 = plt.subplot2grid((5, 4), (4, 3));
        
        Before = [np.mean(self.spreadBefore[int(i):int(i+self.res)]) for i in np.arange(0,len(self.spreadBefore),self.res)];
        ax4.plot(np.arange(0,len(Before)),Before,color=cBefore,alpha=0.5);
        
        After = [np.mean(self.spreadAfter[int(i):int(i+self.res)]) for i in np.arange(0,len(self.spreadAfter),self.res)];
        ax4.plot(np.arange(len(Before),len(Before)+len(After)),After,color=cAfter,alpha=0.5);
        
        ax4.set_title("Cotton spread over time", fontsize = 10);
        ax4.set_ylabel("Cotton spread (px)");
        ax4.set_xlabel("time (h)");
        ax4.set_xticks(np.arange(0,(self._args["plot"]["limit"]*3600)/res,3600/res));
        ax4.set_xticklabels(np.arange(0,self._args["plot"]["limit"]+1,1));
        ax4.set_xlim([0,len(Before+After)])
        
        ax5 = plt.subplot2grid((5, 4), (0, 0), rowspan=5, colspan=3);
        
        ax5.set_xlim([0, int(self.ROIWidth)]);
        ax5.set_ylim([0, int(self.ROILength)]);
        
        self.distTraveledBeforeInitiation = 0;
        self.distTraveledAfterInitiation = 0;
        
        self.posBefore = self._positions[0:int(self._tStartBehav*np.mean(self._framerate))];
        
        if len(self.posBefore) >= self._args["plot"]["limit"]*3600*np.mean(self._framerate) :
            self.posBefore = self._positions[0:int(self._args["plot"]["limit"]*3600*np.mean(self._framerate))];
        
        self.posAfter = self._positions[int(self._tStartBehav*np.mean(self._framerate)):int(self._Length*np.mean(self._framerate))];
        
        if len(self.posBefore)+len(self.posAfter) >= self._args["plot"]["limit"]*3600*np.mean(self._framerate) :
            self.posAfter = self._positions[int(self._tStartBehav*np.mean(self._framerate)):int(self._args["plot"]["limit"]*3600*np.mean(self._framerate))];
        
        self.filteredPosBefore = self.posBefore[0::self._args["plot"]["res"]];
        self.filteredPosAfter = self.posAfter[0::self._args["plot"]["res"]];
        
        if line :
        
            self.befPlot = ax5.plot([x[0] for x in self.filteredPosBefore],[y[1] for y in self.filteredPosBefore],'-o',markersize=1,alpha=alpha,solid_capstyle="butt",color=cBefore,label='Before Initiation');
            self.aftPlot = ax5.plot([x[0] for x in self.filteredPosAfter],[y[1] for y in self.filteredPosAfter],'-o',markersize=1,alpha=alpha,solid_capstyle="butt",color=cAfter,label='After Initiation');
            
        else :
            
            self.befPlot = ax5.scatter([x[0] for x in self.filteredPosBefore],[y[1] for y in self.filteredPosBefore],s=10,alpha=alpha,color=cBefore,label='Before Initiation');
            self.aftPlot = ax5.scatter([x[0] for x in self.filteredPosAfter],[y[1] for y in self.filteredPosAfter],s=10,alpha=alpha,color=cAfter,label='After Initiation');
        
#        else :
#            
#            for x,y in zip([x[0] for x in self.filteredPosBefore],[y[1] for y in self.filteredPosBefore]) :
#                
#                self.befPlot = ax3.plot(x,y,'-',markersize=1,alpha=alpha,solid_capstyle="butt",color='blue',label='Before Initiation');
#                
#            for x,y in zip([x[0] for x in self.filteredPosAfter],[y[1] for y in self.filteredPosAfter]) :
#                
#                self.aftPlot = ax3.plot(x,y,'-',markersize=1,alpha=alpha,solid_capstyle="butt",color='red',label='After Initiation');
        
        self.distTraveledBeforeInitiation = sum(self.distanceCorrected[0:int(self._tStartBehav*np.mean(self._framerate))]);
        
        if len(self.posBefore) >= self._args["plot"]["limit"]*3600*np.mean(self._framerate) :
            self.distTraveledBeforeInitiation = sum(self.distanceCorrected[0:int(self._args["plot"]["limit"]*3600*np.mean(self._framerate))]);
            
        self.distTraveledAfterInitiation = sum(self.distanceCorrected[int(self._tStartBehav*np.mean(self._framerate)):int(self._Length*np.mean(self._framerate))]);
            
        if len(self.posBefore)+len(self.posAfter) >= self._args["plot"]["limit"]*3600*np.mean(self._framerate) :
            self.distTraveledAfterInitiation = sum(self.distanceCorrected[int(self._tStartBehav*np.mean(self._framerate)):int(self._args["plot"]["limit"]*3600*np.mean(self._framerate))]);
                    
        self.distTraveledBeforeInitiation = "%.2f" % (self.distTraveledBeforeInitiation/100);
        self.distTraveledAfterInitiation = "%.2f" % (self.distTraveledAfterInitiation/100);
        
        hB,mB,sB = utils.HoursMinutesSeconds(len(self.posBefore)/np.mean(self._framerate));
        hA,mA,sA = utils.HoursMinutesSeconds(len(self.posAfter)/np.mean(self._framerate));
        
        fontBefore = { "size" : 10,
                      "color" : cBefore,
                      "alpha" : 0.5,
        };
                      
        fontAfter = { "size" : 10,
                      "color" : cAfter,
                      "alpha" : 0.5,
        };
        
        ax5.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
        
        ax5.text(int(self.ROIWidth)/6, -20, 'Time before : {0}h {1}m {2}s'.format(str(hB),str(mB),str(sB)),fontdict=fontBefore);
        ax5.text(int(self.ROIWidth)/6, -10, 'Dist before : {0}'.format(self.distTraveledBeforeInitiation)+'m',fontdict=fontBefore);
        ax5.text(int(self.ROIWidth)/6+150, -20, 'Time after : {0}h {1}m {2}s'.format(str(hA),str(mA),str(sA)),fontdict=fontAfter);
        ax5.text(int(self.ROIWidth)/6+150, -10, 'Dist after : {0}'.format(self.distTraveledAfterInitiation)+'m',fontdict=fontAfter);
        
        if line : 
            ax5.legend(handles = [self.befPlot[0],self.aftPlot[0]],loc=(0,1.02),shadow=True);
            
        else :
            ax5.legend(handles = [self.befPlot,self.aftPlot],loc=(0,1.02),shadow=True);
        
        plt.gca().invert_yaxis();
        
        plt.tight_layout();
        
        if self._args["plot"]["save"] :
            
            plt.savefig(os.path.join(self._args["main"]["resultDir"],"Complete_Tracking_Mouse_{0}".format(self._mouse)));
            
    def OptimizeSegmentation(self, x, peakDist) :
        
        peakDist = peakDist
        
        self.totalTimeAutomatic = 0;
        
        Before = [np.mean(self.cottonBefore[int(i):int(i+self.res)]) for i in np.arange(0,len(self.cottonBefore),self.res)];
        After = [np.mean(self.cottonAfter[int(i):int(i+self.res)]) for i in np.arange(0,len(self.cottonAfter),self.res)];
        All = Before+After;
        
        rasterSpread = (self._args["plot"]["limit"]+1)/(len(All));
        
        #----------------------------------------------------------------------
        #Detection of peaks
        #peakThresh = 0, peakDist = 1, minDist = 2
        
        peaks = [];
        
        for i,data in enumerate(All) :

            if i <= len(All)-peakDist-1 :
            
                if All[i]+x[0] <= All[i+peakDist] or All[i]-x[0] >= All[i+peakDist] :
                    
                    peaks.append(i);
        
        #----------------------------------------------------------------------
        #Merge close peaks
        
        automaticStart = [];
        automaticEnd = [];
        
        if x[1] != None :
        
            for pos,peak in enumerate(peaks) :
                
                if automaticStart == [] :
                    
                    automaticStart.append(peaks[pos]);
                    
                if pos < len(peaks)-1 :
                    
                    if peaks[pos+1]-peaks[pos] >= x[1] : #If next is far 
                        
                        if automaticStart[-1] == peaks[pos] : #If start is the first
                            
                            automaticEnd.append(peaks[pos]+rasterSpread);
                            automaticStart.append(peaks[pos+1]);
                        
                        else :
                            
                            automaticEnd.append(peaks[pos]);
                            automaticStart.append(peaks[pos+1]);
                        
                    else :
                        
                        pass;
                        
            for s,e in zip(automaticStart,automaticEnd) :
                
                self.totalTimeAutomatic += e-s;
                
        else :
            
            for peak in peaks :
                
                self.totalTimeAutomatic += rasterSpread;
        
#        print(x)
#        print([self.totalTimeAutomatic-self.totalTimeManual,0,0])
#        print("\n")
                
        return [self.totalTimeAutomatic,0,0]; #-self.totalTimeManual
                
    def DataManual(self) :
                        
        #----------------------------------------------------------------------
        #Manual raster
        
        self.res = self._framerate*1
        
        self.totalTimeManual = 0;

        path = self._args["main"]["workingDir"]
        
        Before = [np.mean(self.cottonBefore[int(i):int(i+self.res)]) for i in np.arange(0,len(self.cottonBefore),self.res)];
        After = [np.mean(self.cottonAfter[int(i):int(i+self.res)]) for i in np.arange(0,len(self.cottonAfter),self.res)];

        for f in os.listdir(path) :
                    
            path2File = os.path.join(path,f);
            
            if len(f.split(".")) == 2 :
            
                if f.split(".")[-1] == "xlsx" :
                    
                    try :
                    
                        mouse = f.split("-")[1].split(".")[0];
        
                        if mouse == self._args["main"]["mouse"] :
                            
                            excelSheet = pd.read_excel(path2File,header = None);
                            excelMatrix = excelSheet.as_matrix();
                    
                            for line in excelMatrix :
                                
                                if isinstance(line[0], str) :
                                    
                                    temp = [];
                                    d = line[1:];
                                    
                                    for e in d :
                                        
                                        if e >= 0 and not np.isnan(e) :
                                            
                                            temp.append(e);
                                    
                                        if line[0] == "tInjection" :
                                            
                                            tInjection = temp;
                                        
                                        if line[0] == "tCotton" :
                                            
                                            tCotton = temp;
                                            
                                        if line[0] == "tInitiate" :
                                            
                                            tInitiate = temp;
                                            
                                        if line[0] == "tStart" :
                                            
                                            tStart = temp;
                                            
                                        if line[0] == "tEnd" :
                                            
                                            tEnd = temp;
                                            
                                        if line[0] == "tStartInteract" :
                                    
                                            tStartInteract = temp;
                                        
                                        if line[0] == "tEndInteract" :
                                        
                                            tEndInteract = temp;
                                            
                    except :
                        
                        pass;
                                    
        #----------------------------------------------------------------------
        #Detection of peaks
        
        for s,e in zip(tStart,tEnd) :
            
            if s < len(Before+After) :
                
                if e >= len(Before+After) :
                    
                    self.totalTimeManual += len(Before+After)-s;
                    
                elif e < len(Before+After) :
        
                    self.totalTimeManual += e-s;
        
        
        for s,e in zip(tStartInteract,tEndInteract) :
            
            if s < len(Before+After) :
                
                if e >= len(Before+After) :
                    
                    self.totalTimeManual += len(Before+After)-s;
                    
                elif e < len(Before+After) :
        
                    self.totalTimeManual += e-s;
        
            
    def NestingRaster(self, cBefore='b',cAfter='r', res=1, rasterSpread=10, peakThresh = 1, peakDist = 1, minDist = 10) :
        
        fig = plt.figure(figsize=(20,10));
            
        self.res = self._framerate*res
        
        ax0 = plt.subplot(311);
        ax1 = plt.subplot(312,sharex=ax0);
        
        Before = [np.mean(self.cottonBefore[int(i):int(i+self.res)]) for i in np.arange(0,len(self.cottonBefore),self.res)];

        After = [np.mean(self.cottonAfter[int(i):int(i+self.res)]) for i in np.arange(0,len(self.cottonAfter),self.res)];
        
        All = Before+After;
        
#        print(len(All))
        
        peaks = [];
        
        for i,data in enumerate(All) :

            if i <= len(All)-peakDist-1 :
            
                if All[i]+peakThresh <= All[i+peakDist] or All[i]-peakThresh >= All[i+peakDist] :
                    
                    peaks.append(i);
                    
        if rasterSpread == None :
            rasterSpread = (self._args["plot"]["limit"]+1)/(len(All));
            
        self.totalTimeAutomatic = 0;
        
        automaticStart = [];
        automaticEnd = [];
        
        if minDist != None :
        
            for pos,peak in enumerate(peaks) :
                
                if automaticStart == [] :
                    
                    automaticStart.append(peaks[pos]);
                    
                if pos < len(peaks)-1 :
                    
                    if peaks[pos+1]-peaks[pos] >= minDist : #If next is far 
                        
                        if automaticStart[-1] == peaks[pos] : #If start is the first
                            
                            automaticEnd.append(peaks[pos]+rasterSpread);
                            automaticStart.append(peaks[pos+1]);
                        
                        else :
                            
                            automaticEnd.append(peaks[pos]);
                            automaticStart.append(peaks[pos+1]);
                        
                    else :
                        
                        pass;
                        
                else :
                    
                    if automaticStart[-1] == peaks[pos] :
                        
                        automaticEnd.append(peaks[pos]+rasterSpread);
                    
                    else :
                        
                        automaticEnd.append(peaks[pos]);
            
            raster = np.zeros_like(All)
            #raster = [0 for x in np.arange(automaticStart[0])];
            #pos = 0;
            
            for s,e in zip(automaticStart,automaticEnd) :
                
#                if pos+1 < len(automaticStart):
#            
#                    [raster.append(1) for x in np.arange((e-s))];
#                    [raster.append(0) for x in np.arange((automaticStart[pos+1]-e))];
#                    pos+=1;
#                    
#                else :
#                    
#                    [raster.append(1) for x in np.arange((e-s))];
#                    [raster.append(0) for x in np.arange(len(All)-int(automaticEnd[-1]+1))];
                
                if e-s < 1 :
                    
                    raster[int(s):int(e)+1] = 1;
                    
                else :
                
                    raster[int(s)+1:int(e)+1] = 1;
                
                ax0.add_patch(patches.Rectangle((s, 0), e-s, 1,color=cAfter,alpha=1));
                self.totalTimeAutomatic += e-s;
                
        else :
            
            for peak in peaks :
                
                ax0.add_patch(patches.Rectangle((peak, 0), rasterSpread, 1,color=cAfter,alpha=1));
                self.totalTimeAutomatic += rasterSpread;
        
#        print(len(automaticStart))
#        print(len(automaticEnd))
        
#        raster = [0 for x in np.arange(automaticStart[0])];
#        pos = 0;
#                
#        for s,e in zip(automaticStart,automaticEnd) :
#            
#            if pos+1 < len(automaticStart):
#            
#                [raster.append(1) for x in np.arange((e-s))];
#                [raster.append(0) for x in np.arange((automaticStart[pos+1]-e))];
#                pos+=1;
#                
#            else :
#                
#                [raster.append(1) for x in np.arange((e-s))];
#                [raster.append(0) for x in np.arange(len(All)-int(automaticEnd[-1]+1))];
        
#        print(automaticStart)
#        print(automaticEnd)
#        print(len(raster))
        np.save(os.path.join(self._args["main"]["resultDir"],"Data_{0}_Raster.npy".format(self._args["main"]["mouse"])),np.array(raster));
                
                
            
#        print(automaticStart)
#        print(automaticEnd)
#         
#        print(automaticEnd[-1])
#        print(len(raster))
#        print(len(All))

#        print(len(raster))
            
#        ax0.plot(raster,drawstyle="steps")
            
        ax0.set_title("Nest-building activity over time", fontsize = 10);
        ax0.set_xticks(np.arange(0,(self._args["plot"]["limit"]*3600)/res,3600/res));
        ax0.set_xticklabels(np.arange(0,self._args["plot"]["limit"]+1,1));
        ax0.set_xlim([0,len(Before+After)]);
        ax0.set_ylabel("Nest-building activity");
        ax0.tick_params(left=False,labelleft=False,bottom=False,labelbottom=False);
        
        Before = [np.mean(self.cottonBefore[int(i):int(i+self.res)]) for i in np.arange(0,len(self.cottonBefore),self.res)];
        ax1.plot(np.arange(0,len(Before)),Before,color=cBefore,alpha=0.5);
        After = [np.mean(self.cottonAfter[int(i):int(i+self.res)]) for i in np.arange(0,len(self.cottonAfter),self.res)];
        ax1.plot(np.arange(len(Before),len(Before)+len(After)),After,color=cAfter,alpha=0.5);
        
        ax1.set_title("Cotton height over time", fontsize = 10);
        ax1.set_ylabel("Average pixel intensity *8bit (a.u)");
        ax1.set_xticks(np.arange(0,(self._args["plot"]["limit"]*3600)/res,3600/res));
        ax1.set_ylim([100,150]);
        ax1.tick_params(bottom=False,labelbottom=False);
        ax1.set_xlim([0,len(Before+After)]);
        
#        print(len(raster)) 
#        print(len(Before+After))
        
        self.totalTimeManual = 0;

        path = self._args["main"]["workingDir"]

        for f in os.listdir(path) :
                    
            path2File = os.path.join(path,f);
            
            if len(f.split(".")) == 2 :
            
                if f.split(".")[-1] == "xlsx" :
                    
                    try :
                    
                        mouse = f.split("-")[1].split(".")[0];
        
                        if mouse == self._args["main"]["mouse"] :
                            
                            excelSheet = pd.read_excel(path2File,header = None);
                            excelMatrix = excelSheet.as_matrix();
                    
                            for line in excelMatrix :
                                
                                if isinstance(line[0], str) :
                                    
                                    temp = [];
                                    d = line[1:];
                                    
                                    for e in d :
                                        
                                        if e >= 0 and not np.isnan(e) :
                                            
                                            temp.append(e);
                                    
                                    if line[0] == "tInjection" :
                                        
                                        tInjection = temp;
                                    
                                    if line[0] == "tCotton" :
                                        
                                        tCotton = temp;
                                        
                                    if line[0] == "tInitiate" :
                                        
                                        tInitiate = temp;
                                        
                                    if line[0] == "tStart" :
                                        
                                        tStart = temp;
                                        
                                    if line[0] == "tEnd" :
                                        
                                        tEnd = temp;
                                        
                                    if line[0] == "tStartInteract" :
                                
                                        tStartInteract = temp;
                                    
                                    if line[0] == "tEndInteract" :
                                    
                                        tEndInteract = temp;
                                        
#                                    if line[0] == "tStartDig" :
#                            
#                                        tStartDig = temp;
#                                        
#                                    if line[0] == "tEndDig" :
#                                        
#                                        tEndDig = temp;
#                                        
#                                    if line[0] == "tStartGroom" :
#                            
#                                        tStartGroom = temp;
#                                        
#                                    if line[0] == "tEndGroom" :
#                                        
#                                        tEndGroom = temp;
#                                        
                    except :
                            
                        pass;
                            
        ax2 = plt.subplot(313,sharex=ax0)
        temp =0;
        
        for s,e in zip(tStart,tEnd) :
            
            if s < len(Before+After) :
                
                if e >= len(Before+After) :
                    
                    temp+=len(Before+After)-s;
                    ax2.add_patch(patches.Rectangle((s-tCotton[0], 0),len(Before+After)-s,10,alpha = 0.5,fc="blue",ec='None',label="Nesting"));
                    self.totalTimeManual += len(Before+After)-s;
                    
                elif e < len(Before+After) :
                    
                    temp+=e-s;
                    ax2.add_patch(patches.Rectangle((s-tCotton[0], 0),e-s,10,alpha = 0.5,fc="blue",ec='None',label="Nesting"));
                    self.totalTimeManual += e-s;
        
        temp =0;
        
        for s,e in zip(tStartInteract,tEndInteract) :
            
            if s < len(Before+After) :
                
                if e >= len(Before+After) :
                    
                    temp+=len(Before+After)-s;
                    ax2.add_patch(patches.Rectangle((s-tCotton[0], 0),len(Before+After)-s,10,alpha = 0.5,fc="blue",ec='None',label="Cotton Interaction"));
                    self.totalTimeManual += len(Before+After)-s;
                    
                elif e < len(Before+After) :
                    
                    temp+=e-s;
                    ax2.add_patch(patches.Rectangle((s-tCotton[0], 0),e-s,10,alpha = 0.5,fc="blue",ec='None',label="Cotton Interaction"));
                    self.totalTimeManual += e-s;
    
        ax2.set_xlim(0,len(Before+After));
        ax2.set_title("Nest-building activity over time", fontsize = 10);
        ax2.set_ylabel("Nest-building activity");
        ax2.set_xlabel("time (h)");       
        ax2.tick_params(left=False,labelleft=False);           
            
        multi = MultiCursor(fig.canvas, (ax0, ax1, ax2), color='r', lw=2, horizOn=False, vertOn=True)
        
        self.dif = self.totalTimeAutomatic/self.totalTimeManual;
        self.percentage = round((1-abs(1-self.dif))*100,3)
        
        textstr = '\n'.join((
            "Auto = {0}s".format(round(self.totalTimeAutomatic,2)),
            "Manual = {0}s".format(round(self.totalTimeManual,2)),
            "{0}% accuracy".format(self.percentage),
            r"$\alpha$ = {0}".format(peakThresh),
            r"$\beta$ = {0}".format(minDist),));
        
        props = dict(boxstyle='round', facecolor='white', alpha=1)

        # place a text box in upper left in axes coords
        ax0.text(0.01, 0.95, textstr, transform=ax0.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout();
        plt.show()
        
        print(self.totalTimeManual,self.totalTimeAutomatic)
        
    #    if self._args["plot"]["save"] :
    #        
    #        plt.savefig(os.path.join(self._args["main"]["resultDir"],"Raster_Mouse_{0}".format(self._mouse)))
        
    def HeatMapPlot(self,bins=1000,sigma=16) :
        
        fig = plt.figure(figsize=(20,10));
        ax0 = plt.subplot();

        ax0.set_xlim([0, int(self.ROIWidth)]);
        ax0.set_ylim([0, int(self.ROILength)]);
        
        self._x = [x[0] for x in self._positions]+[0,int(self.ROIWidth)];
        self._y = [x[1] for x in self._positions]+[int(self.ROILength),0];
        
        #cmap = plt.get_cmap('jet');
        #norm = mcolors.LogNorm();
        
        heatmap, xedges, yedges = np.histogram2d(self._x, self._y, bins=bins);
        heatmap = gaussian_filter(heatmap, sigma=sigma);

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        ax0.imshow(heatmap.T, extent=extent, origin='lower', cmap=cm.jet)
        
        #ax0.hexbin(self._x, self._y, gridsize=gridsize, cmap=cmap, norm=norm);
        ax0.set_title('Tracking Mouse {0}'.format(self._mouse));
        ax0.tick_params(top=False, bottom=False, left=False, right=False);
        
        plt.gca().invert_yaxis();
        
        