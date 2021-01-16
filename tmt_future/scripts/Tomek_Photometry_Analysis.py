#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:30:37 2019

@author: thomas.topilko
"""

import os;
import pandas as pd;
import matplotlib.pyplot as plt;
import matplotlib.animation as animation;
import matplotlib.patches as patches;
import numpy as np;
import datetime;
from matplotlib.widgets import MultiCursor;

import moviepy.editor as mpy;
from moviepy.video.io.bindings import mplfig_to_npimage;
from moviepy.editor import VideoFileClip, VideoClip, clips_array,ImageSequenceClip;

TMTDir = "/home/thomas.topilko/Documents/TopMouseTracker";
if not os.getcwd() == TMTDir :
    
    os.chdir(TMTDir);
    
import scripts.Tomek_Photometry_Functions as fc;

experiment = 200917;
mouse = "2";

workDir = "/raid/Tomek/Photometry/{0}/{1}".format(experiment, mouse);

photometryFileCSV = os.path.join(workDir, "{0}_{1}.csv".format(experiment, mouse));
#videoFile = os.path.join(workDir, "12-2-2020_15-38-35/Raw_Video_Mice_813_12-2-2020_15-38-35.avi");
#cottonFile = os.path.join(workDir, "{0}/Data_{0}_CottonPixelIntensities.npy".format(mouse));
manualFile = os.path.join(workDir, "{0}_{1}_manual.xlsx".format(experiment, mouse));

#%%
###############################################################################
#Reading the Photometry data
###############################################################################

print("\n");
print("Reading Photometry data");

""" File structure :
    -----------------------------------
    | Time(s) | Dem Out-1 | Dem Out-2 |
    -----------------------------------
"""

DF = 50.; #Decimation factor for data downsampling
SRD = 12000./DF; #Sampling rate of the Doric system
rollAvg = 100; #Rolling Average Parameter (s)
startVideo = 0*3600+0*60+0; #When to start the video
endVideo = 0*3600+21*60+13; #When to end the video
videoLength = 0*3600+21*60+13; #The length of the video ;; FOR TIME COMPRESSION

f = photometryFileCSV; #photometryFileCSV
photometryFileNPY = fc.ConvertPhotometryData(f); #Convert CSV file to NPY if needed photometryFileCSV

#fc.checkTimeShift(photometryFileNPY, SRD, plot=True, label="Decimation = 1 ; Video");

args = {"lw" : 1., #linewidth in plot
        "fsl" : 12., #fontsize for labels
        "fst" : 16., #fontsize for titles
        "save" : False, #saving intermediate plots
        "function" : "moving_avg", #low_pass moving_avg None ---- To filter the dF signal
        "freq" : 2, #frequency for low pass filter
        "order" : 10, #order of the low pass filter
        "roll_size" : 200, #rolling average parameter
        "which" : [False, False, False, False, False, True] #Raw, Corrected, Standardized, Fit, Aligned, dF
        };

rawX, rawIsosbestic, rawCalcium, dF = fc.LoadPhotometryData(photometryFileNPY, SRD,\
                                                            startVideo, endVideo, videoLength, rollAvg, optPlots=True,\
                                                            recompute=True, plotargs=args); #Loads the NPY data
                                                       
#%%
###############################################################################
#Reading the Video data
###############################################################################

print("\n");
print("Reading Video data");

videoClip = mpy.VideoFileClip(videoFile).subclip(t_start=startVideo, t_end=endVideo);
#plt.imshow(videoClip.get_frame(0));

videoClipCropped = videoClip.crop(x1=763, y1=264, x2=1288, y2=526);
#plt.imshow(videoClipCropped.get_frame(0));

fps = videoClipCropped.fps;

#%%
###############################################################################
#Reading the Cotton data from automatic segmentation
###############################################################################

print("\n");
print("Reading Cotton data");

rawCotton, Cotton = fc.LoadTrackingData(cottonFile, fps, startVideo, endVideo);

"""Peak Detection"""
PAT = 0.7; #Peak Amplitude Threshold
PTPD = 1; #Peak to Peak Distance
PMD = 7.; #Peak Merging Distance
RS = 1./fps; #Raster Spread

peaks, posPeaks = fc.DetectRawPeaks(Cotton, PAT, PTPD);
peaksMerged, peaksPosMerged  = fc.MergeClosePeaks(peaks, posPeaks, PMD);

"""Major peak detection"""
DBE = 2; #Distance Between Events 2
BL = 0; #Behavioral Bout Length
GD = 40.; #Graph distance (distance before+after the event started)

mPeaks, mPosPeaks, mLengthPeaks, seeds, posSeeds = fc.DetectMajorBouts(peaksMerged, DBE, BL, GD);

#fc.PeakPlot(posPeaks, peaksPosMerged, mPosPeaks, posSeeds, multiPlot=True, timeFrame=[], save=False);

#%%
###############################################################################
#Reading the Cotton data from manual segmentation
###############################################################################

"""Peak Detection"""
PMD = 7.; #Peak Merging Distance

peaks, posPeaks = fc.extract_manual_bouts(manualFile, endVideo, "Nesting")
#peaksMerged = fc.extractManualBehavior(Cotton, manualFile);
peaksMerged, peaksPosMerged  = fc.MergeClosePeaks(peaks, posPeaks, PMD);

"""Major peak detection"""
DBE = 2; #Distance Between Events 2
BL = 0; #Behavioral Bout Length
GD = 40.; #Graph distance (distance before+after the event started)

mPeaks, mPosPeaks, mLengthPeaks, seeds, posSeeds = fc.DetectMajorBouts(peaksMerged, DBE, BL, GD);

fc.PeakPlot(posPeaks, peaksPosMerged, mPosPeaks, posSeeds, multiPlot=True, timeFrame=[], save=False);

#%%
###############################################################################
#Reading the Cotton data from manual segmentation
###############################################################################

"""Extracting Calcium data from the major peak detection zone"""

#fc.savePeriEventData(dataAroundPeaks, mLengthPeaks, mPosPeaks, workDir, mouse);

boutThresh = 1; #Behavioral Bout Length Threshold for display
GD = 40.; #Graph distance (distance before+after the event started)

dataAroundPeaks = fc.ExtractCalciumDataWhenBehaving(mPosPeaks, dF, GD, SRD, resample=1)
dataAroundPeaksFiltered, mLengthPeaksFiltered, mPosPeaksFiltered = fc.filterShortBouts(dataAroundPeaks, mLengthPeaks, mPosPeaks, boutThresh);
dataAroundPeaksOrdered, mLengthPeaksOrdered, mPosPeaksOrdered = fc.reorderByBoutSize(dataAroundPeaksFiltered, mLengthPeaksFiltered, mPosPeaksFiltered);
GDb, GDf = GD, GD;

#cmap = gnuplot, twilight, inferno, viridis

#fig=plt.figure(figsize=(10,7))
#ax0 = plt.subplot(111)
#ax0.plot(dF)
#Min = min(dF)
#Max = max(dF)
#for i, j in zip(np.array(mPosPeaks)*SRD, np.array(mLengthPeaks)*SRD) :
#    patch = patches.Rectangle((i,Min), j, Max, alpha=0.5, color="red")
#    ax0.add_patch(patch)
#    ax0.plot([i,i], [Min, Max], color="red")

fc.PeriEventPlot(dataAroundPeaksOrdered, mLengthPeaksOrdered, SRD, SRD, GD, GDf, GDb, SRD, save=False,\
                 showStd=True, fileNameLabel="_All", cmap="inferno", lowpass=(2,2));

#%%

def LivePhotometryTrack(vidClip, x, yData, thresh, globalAcceleration=1,\
                        plotAcceleration=SRD, showHeight=True, showTrace=True) :
    
    nSubplots = len([x for x in yData if x != []]);
    
    def make_frame_mpl(t):
         
        i = int(t);
        
        if i < thresh*plotAcceleration :
            
            try :
                
                cottonGraph.set_data(x[0:i], yData[0][0:i]);
#                cottonGraph.fill_between(x[0:i], 0, y0[0:i], color="blue", alpha=0.8)
#                eventGraph.set_data(x[0:i], y1);
#                calciumGraph.set_data(x[0:i], y2[0:i]);
                dFGraph.set_data(x[0:i], yData[3][0:i]);
                
            except :
                
                print("Oups a problem occured");
                pass;
    
            last_frame = mplfig_to_npimage(liveFig);
            return last_frame;
        
        else :
            
            delta = (i/plotAcceleration) - thresh;
        
            liveAx0.set_xlim(x[0]+delta, x[0]+(i/plotAcceleration));
#            liveAx1.set_xlim(x[0]+delta, x[0]+(i/plotAcceleration));
#            liveAx2.set_xlim(x[0]+delta, x[0]+(i/plotAcceleration));
            liveAx3.set_xlim(x[0]+delta, x[0]+(i/plotAcceleration));
            
            try :
                
                cottonGraph.set_data(x[0:i], yData[0][0:i]);
#                cottonGraph.fill_between(x[0:i], 0, y0[0:i], color="blue", alpha=0.8)
#                eventGraph.set_data(x[0:i], y1[0:i]);
#                calciumGraph.set_data(x[0:i], y2[0:i]);
                dFGraph.set_data(x[0:i], yData[3][0:i]);
                
            except :
                
                print("Oups a problem occured");
                pass;
    
            last_frame = mplfig_to_npimage(liveFig);
            return last_frame;
    
    _FrameRate = vidClip.fps;
    _Duration = vidClip.duration;
    
    liveFig = plt.figure(figsize=(10,6), facecolor='white');
    
    gs = liveFig.add_gridspec(nrows=nSubplots, ncols=1);
    
    #First live axis
    liveAx0 = liveFig.add_subplot(gs[0, :]);
    
    liveAx0.set_title("Cotton height");
    liveAx0.set_xlim([x[0], x[0]+thresh]);
    liveAx0.set_ylim([min(yData[0])-(min(yData[0])*0.05), max(yData[0])+(max(yData[0])*0.05)]);
    
    cottonGraph, = liveAx0.plot(x[0], yData[0][0], '-', color="blue", alpha=0.8, ms=1.);
    
#    #Second live axis
#    liveAx1 = liveFig.add_subplot(gs[1, :]);
#    
#    liveAx1.set_title("Raster plot");
#    liveAx1.set_xlim([x[0], x[0]+thresh]);
#    liveAx1.set_ylim([-1, 1]);
#    
#    eventGraph, = liveAx1.eventplot(y1,colors="blue",alpha=0.8);
#    
    #Third live axis
#    liveAx2 = liveFig.add_subplot(gs[0, :]);
#    
#    liveAx2.set_title("Calcium dependant trace (465nm) (mV)");
#    liveAx2.set_xlim([x[0], x[0]+thresh]);
#    liveAx2.set_ylim([min(y2)-(min(y2)*0.1), max(y2)+(max(y2)*0.1)]);
#    
#    calciumGraph, = liveAx2.plot(x[0], y2[0], '-', color="cyan", alpha=0.8, ms=1., lw=0.5);
    
    #Fourth live axis
    liveAx3 = liveFig.add_subplot(gs[1, :]);
    
    liveAx3.set_title("dF/F");
    liveAx3.set_xlim([x[0], x[0]+thresh]);
    liveAx3.set_ylim([min(yData[3])-(min(yData[3])*0.05), max(yData[3])+(max(yData[3])*0.05)]);

    dFGraph, = liveAx3.plot(x[0], yData[3][0], '-', color="green", alpha=0.8, ms=1., lw=0.5);
    
    plt.tight_layout();
    
    anim = mpy.VideoClip(make_frame_mpl, duration=(_Duration*plotAcceleration));
    
    _clips = [clip.margin(2, color=[255,255,255]) for clip in [(vidClip.resize(2.).speedx(globalAcceleration)), anim.speedx(globalAcceleration).speedx(plotAcceleration)]];
    
    finalClip = clips_array([[_clips[0]],
                             [_clips[1]]],
                             bg_color=[255,255,255]);
    
    finalClip.write_videofile(os.path.join("/home/thomas.topilko/Desktop",'PhotoMetry_Tracking.mp4'), fps=10);
    


#rawXZeroed = rawX - rawX[0]
newCotton = [np.full(240, i) for i in peaksMerged];
newCotton = [x for sublist in newCotton for x in sublist];
#newCotton.append(newCotton[-1]);
newCotton = np.array(newCotton);

yData = [newCotton, [], [], dF];

for y in yData :
    
    print(len(y))

displayThreshold = int(5*videoClipCropped.fps); #How much of the graph will be displayed in live (x scale)

LivePhotometryTrack(videoClipCropped, rawX, yData, displayThreshold,\
                    globalAcceleration=5, plotAcceleration=SRD);

#%%
###############################################################################
#Main Peri Event plot
###############################################################################

peFolder = "/raid/Tomek/Photometry/peData";

boutThresh = 10;
GDb, GDf = 5, 40;

peData = [];
leData = [];
posData = [];

for f in os.listdir(peFolder) :
    
    temp = np.load(os.path.join(peFolder, f));
    peData.append(temp[0]);
    leData.append(temp[1]);
    posData.append(temp[2]);
    
peData = [x for sublist in peData for x in sublist];
leData = [x for sublist in leData for x in sublist];
posData = [x for sublist in posData for x in sublist];

peData, leData, posData = fc.filterShortBouts(peData, leData, posData, boutThresh);
peData, leData, posData = fc.reorderByBoutSize(peData, leData, posData);
    
fc.PeriEventPlot(peData, leData, SRD, SRD, GD, GDf, GDb, SRD, save=False,\
                 showStd=True, fileNameLabel="_All", cmap="inferno",\
                 lowpass=(2,2), norm=True)                  
                    