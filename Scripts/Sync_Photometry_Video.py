#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:21:34 2019

@author: thomas.topilko
"""

import os;
import pandas as pd;
import matplotlib.pyplot as plt;
import matplotlib.animation as animation;
import numpy as np;

import moviepy.editor as mpy;
from moviepy.video.io.bindings import mplfig_to_npimage;
from moviepy.editor import VideoFileClip, VideoClip, clips_array,ImageSequenceClip;

photometryFile = "/mnt/raid/TopMouseTracker/191119/Trace_1.csv";
videoFile = "/mnt/raid/TopMouseTracker/191119/19-11-2019_11-42-23/Raw_Video_Mice_808_19-11-2019_11-42-23.avi";
cottonFile = "/mnt/raid/TopMouseTracker/191119/808/Data_808_CottonPixelIntensities.npy";

###############################################################################
#Reading the Photometry data
###############################################################################
print("\n");
print("Reading Photometry data");

samplingRateDoric = 12000.; #Sampling rate of the Doric system
displayResolution = 1.; #Resolution for the display : 1 = 1s/point

startVideo = 0; #When to start the video
endVideo = 3*3600+19*60+35; #When to end the video

if "photometry_data.npy" in os.listdir( os.path.dirname(photometryFile) ) :
    
    photometryData = np.transpose( np.load( os.path.join(os.path.dirname(photometryFile), "photometry_data.npy" )) );
    
else :

    photometrySheet = pd.read_csv( photometryFile, header=1, usecols=np.arange(0,3) ); #Load the data
    nonNanMask = photometrySheet["AIn-1 - Dem (AOut-1)"] > 0.; #Filtering NaN values (missing values)
    filteredPhotometrySheet = photometrySheet[nonNanMask]; #Filter the data
    photometryDataNumpy = filteredPhotometrySheet.to_numpy(); #Convert to numpy for speed
    np.save( os.path.join( os.path.dirname(photometryFile), "photometry_data.npy"), photometryDataNumpy ); #Save the data as numpy file
    photometryData = np.transpose(photometryDataNumpy);
    
rawX = np.array([ np.mean( [  float(i) for i in photometryData[0][:][ int(n) : int(n+(samplingRateDoric*displayResolution)) ] ] )\
                 for n in np.arange(0, len(photometryData[0][:]), (samplingRateDoric*displayResolution)) ]);
    
rawIsosbestic = np.array([ np.mean( [  float(i) for i in photometryData[1][:][ int(n) : int(n+(samplingRateDoric*displayResolution)) ] ] )\
                          for n in np.arange(0, len(photometryData[1][:]), (samplingRateDoric*displayResolution)) ]);
    
rawCalcium = np.array([ np.mean( [  float(i) for i in photometryData[2][:][ int(n) : int(n+(samplingRateDoric*displayResolution)) ] ] )\
                           for n in np.arange(0, len(photometryData[2][:]), (samplingRateDoric*displayResolution)) ]);
    
X = rawX[startVideo:endVideo];
X = X - X[0];

Isosbestic = rawIsosbestic[startVideo:endVideo];
Calcium = rawCalcium[startVideo:endVideo];

###############################################################################
#Reading the Video data
###############################################################################
print("\n");
print("Reading Video data");

videoClip = mpy.VideoFileClip(videoFile).subclip(t_start=startVideo, t_end=endVideo).crop(x1=870,y1=494,x2=1401,y2=741);

displayThreshold = int(50*videoClip.fps); #How much of the graph will be displayed in live (x scale)

###############################################################################
#Reading the Cotton data
###############################################################################
print("\n");
print("Reading Cotton data");

cottonData = np.load(cottonFile); #Read the file

rawCotton = np.array([ np.mean( [  float(i) for i in cottonData[ int(n) : int(n+(videoClip.fps)) ] ] )\
                      for n in np.arange( 0, len(cottonData), (videoClip.fps) ) ]);
    
Cotton = rawCotton[:len(rawX)]; #Match the array length
Cotton = Cotton[startVideo:endVideo]; #Crop in time

###############################################################################
#Peak Detection
###############################################################################
peakAmplitudeThreshold = 0.7;
peakToPeakDistance = 1;
peakMergingDistance = 7.;
rasterSpread = 1./videoClip.fps;

detectedPeaks = [];
posDetectedPeaks = [];

#Raw peack detection
print("\n");
print("Running peak detection algorithm");

for i, dp in enumerate(Cotton) : #For every data point
    
    if i <= (len(Cotton) - 1) - peakToPeakDistance : #If the point has enough points available in front of it
        
        if Cotton[i]+peakAmplitudeThreshold <= Cotton[i+peakToPeakDistance]\
        or Cotton[i]-peakAmplitudeThreshold >= Cotton[i+peakToPeakDistance] :
            
            detectedPeaks.append(True);
            posDetectedPeaks.append(i);
            
        else :
            
            detectedPeaks.append(False);
            
#Merge close peaks
print("\n");
print("Running peak merging algorithm");

detectedPeaksMerged = [];

for pos, peak in enumerate(detectedPeaks) :
    
    if peak :
            
        ind = posDetectedPeaks.index(pos);
        
        if ind < len(posDetectedPeaks) - 1 :
        
            if posDetectedPeaks[ind+1] - posDetectedPeaks[ind] < peakMergingDistance :
                
                for i in np.arange(0, posDetectedPeaks[ind+1] - posDetectedPeaks[ind]) :
                    
                    detectedPeaksMerged.append(True);
        
        else :
            
            detectedPeaksMerged.append(True);
                
    else :
        
        if len(detectedPeaksMerged) <= pos :
        
            detectedPeaksMerged.append(False);

def LivePhotometryTrack(vidClip, x, y0, y1, y2, y3, thresh, globalAcceleration=1,\
                        plotAcceleration=1./displayResolution, showHeight=True, showTrace=True) :
    
    def make_frame_mpl(t):
         
        i = int(t);
        
        if i < thresh :
            
            try :
                
                cottonGraph.set_data(x[0:i],y0[0:i]);
                eventGraph.set_data(x[0:i],y1[0:i]);
                calciumGraph.set_data(x[0:i],y2[0:i]);
                isosbesticGraph.set_data(x[0:i],y3[0:i]);
                
            except :
                
                print("Oups a problem occured")
                pass;
    
            last_frame = mplfig_to_npimage(liveFig);
            return last_frame;
        
        else :
            
            delta = i - thresh;
        
            liveAx0.set_xlim(delta,i);
            liveAx1.set_xlim(delta,i);
            liveAx2.set_xlim(delta,i);
            liveAx3.set_xlim(delta,i);
            
            try :
                
                cottonGraph.set_data(x[0:i],y0[0:i]);
                eventGraph.set_data(x[0:i],y1[0:i]);
                calciumGraph.set_data(x[0:i],y2[0:i]);
                isosbesticGraph.set_data(x[0:i],y3[0:i]);
                
            except :
                
                print("Oups a problem occured")
                pass;
    
            last_frame = mplfig_to_npimage(liveFig);
            return last_frame;
    
#    _FrameRate = vidClip.fps;
    _Duration = vidClip.duration;
    
    liveFig = plt.figure(figsize=(10,6), facecolor='white');
    
    gs = liveFig.add_gridspec(ncols=1, nrows=4);
    
    #First live axis
    liveAx0 = liveFig.add_subplot(gs[0, :]);
    
    liveAx0.set_title("Cotton height");
    liveAx0.set_xlim([0, thresh]);
    liveAx0.set_ylim([min(y0), max(y0)]);
    
    cottonGraph, = liveAx0.plot(x[0],y0[0],'-',color="blue",alpha=0.8,ms=1.);
    
    #Second live axis
    liveAx1 = liveFig.add_subplot(gs[1, :]);
    
    liveAx1.set_title("Raster plot");
    liveAx1.set_xlim([0, thresh]);
    liveAx1.set_ylim([-0.1, 1.1]);
    
    eventGraph, = liveAx1.plot(x[0],y1[0],'-',color="blue",alpha=0.8,ms=1.);
    
    #Third live axis
    liveAx2 = liveFig.add_subplot(gs[2, :]);
    
    liveAx2.set_title("Calcium dependant trace (465nm) (mV)");
    liveAx2.set_xlim([0, thresh]);
    liveAx2.set_ylim([0.03, 0.08]);
    
    calciumGraph, = liveAx2.plot(x[0],y2[0],'-',color="red",alpha=0.8,ms=1.);
    
    #Fourth live axis
    liveAx3 = liveFig.add_subplot(gs[3, :]);
    
    liveAx3.set_title("Isosbestic trace (405nm) (mV)");
    liveAx3.set_xlim([0, thresh]);
    liveAx3.set_ylim([0.063, 0.073]);

    isosbesticGraph, = liveAx3.plot(x[0],y3[0],'-',color="green",alpha=0.8,ms=1.);
    
#    plt.gca().invert_yaxis();
#        plt.gca().invert_xaxis();
    plt.tight_layout();
    
#    a = make_frame_mpl(1000);
#    return a;
    
    anim = mpy.VideoClip(make_frame_mpl, duration=(_Duration*1));
    
    finalClip = clips_array([[clip.margin(2, color=[255,255,255]) for clip in
                    [(vidClip.resize(2.).speedx(globalAcceleration)), anim.speedx(globalAcceleration).speedx(plotAcceleration)]]],
                    bg_color=[255,255,255]); #.speedx(self._FrameRate)
    
    finalClip.write_videofile(os.path.join("/home/thomas.topilko/Desktop",'PhotoMetry_Tracking.mp4'), fps=10);
    
LivePhotometryTrack(videoClip, X, Cotton, detectedPeaksMerged, Calcium, Isosbestic,\
                    displayThreshold, globalAcceleration=5, plotAcceleration=1./displayResolution);
                 
                    
boutDist = 50;

def BehaviorPhotometryPlot(peaks, boutDist) :
    
    initialPeaks = [];
    posInitialPeaks = [];
    detectedPeak = False;
    distToNextPeak = 0;
    cumulativeBoutToNextPeak = 0;
    
    for pos, peak in enumerate(peaks) :
        
        if detectedPeak :
            
            distToNextPeak += 1;
        
        if peak :
            
            cumulativeBoutToNextPeak += 1;
            detectedPeak = True;
            
            if not True in initialPeaks :
                    
                initialPeaks.append(True);
                posInitialPeaks.append(pos);
                
            if distToNextPeak > boutDist :
                
                initialPeaks.append(True);
                posInitialPeaks.append(pos);
                distToNextPeak = 0;
                
#                print(cumulativeBoutToNextPeak)
                
                if cumulativeBoutToNextPeak < 15 :
                    
                    print(cumulativeBoutToNextPeak)
                    
                    initialPeaks = initialPeaks[:-1];
                    initialPeaks.append(False);
                    posInitialPeaks = posInitialPeaks[:-1];
                
                cumulativeBoutToNextPeak = 0;
                
            else :
                
                initialPeaks.append(False);
                distToNextPeak = 0;
                
        else :
            
            initialPeaks.append(False);
                
    return posInitialPeaks, initialPeaks;

posInitialPeaks, initialPeaks = BehaviorPhotometryPlot(detectedPeaksMerged, boutDist);

fig = plt.figure();
ax0 = plt.subplot(111);
ax0.plot(detectedPeaksMerged)
ax0.plot(initialPeaks)

def ExtractCalciumDataWhenBehaving(posPeaks, calciumData, boutDist) :
    
    data = [];
    
    for p in posPeaks :   
        
        data.append(calciumData[p-boutDist : p+boutDist+1]);
        
    return data;
              
calciumDataAroundPeaks = ExtractCalciumDataWhenBehaving(posInitialPeaks, Calcium, boutDist);

Mean = np.mean(calciumDataAroundPeaks, axis=0);
Std = np.std(calciumDataAroundPeaks, axis=0);

fig = plt.figure();
ax0 = plt.subplot(211);
ax0.plot(np.arange(-boutDist, boutDist+1, 1), Mean, color="blue", alpha=0.5);
ax1 = plt.subplot(212);
#ax1.set_xlim(-boutDist, boutDist+1, 1));
ax1.imshow(calciumDataAroundPeaks, cmap='viridis', interpolation='nearest');
#ax0.plot(np.arange(-boutDist, boutDist+1, 1), Mean+Std, color="blue", alpha=0.5);
#ax0.plot(np.arange(-boutDist, boutDist+1, 1), Mean-Std, color="blue", alpha=0.5);
#ax0.fill_between(np.arange(-boutDist, boutDist+1, 1), Mean, Mean+Std, color="blue", alpha=0.1);
#ax0.fill_between(np.arange(-boutDist, boutDist+1, 1), Mean, Mean-Std, color="blue", alpha=0.1);
        
    
    
