#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:21:34 2019

@author: thomas.topilko
"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoFileClip, VideoClip, clips_array, ImageSequenceClip


workDir = "/home/thomas.topilko/Desktop/Photometry/191203";

photometryFile = "/home/thomas.topilko/Desktop/Photometry/200212/813_50df_1.csv";
#dFFile = "/home/thomas.topilko/Desktop/dF.npy";
videoFile = "/home/thomas.topilko/Desktop/Photometry/200212/813/Tracking_813.avi";
#cottonFile = "/mnt/raid/TopMouseTracker/191119/808/Data_808_CottonPixelIntensities.npy";

###############################################################################
#Reading the Photometry data
###############################################################################

print("\n");
print("Reading Photometry data");

samplingRateDoric = 12000.; #Sampling rate of the Doric system
decimationFactor = 50.
ratio = samplingRateDoric/decimationFactor
displayResolution = 1.; #Resolution for the display : 1 = 1s/point

startVideo = 0; #When to start the video
endVideo = 0*3600+10*60+0; #When to end the video

#if "photometry_data.npy" in os.listdir( os.path.dirname(photometryFile) ) :
#    
#    photometryData = np.transpose( np.load( os.path.join(os.path.dirname(photometryFile), "photometry_data.npy" )) );
#    
#else :
#
#    photometrySheet = pd.read_csv( photometryFile, header=1, usecols=np.arange(0,3) ); #Load the data
#    nonNanMask = photometrySheet["AIn-1 - Dem (AOut-1)"] > 0.; #Filtering NaN values (missing values)
#    filteredPhotometrySheet = photometrySheet[nonNanMask]; #Filter the data
#    photometryDataNumpy = filteredPhotometrySheet.to_numpy(); #Convert to numpy for speed
#    np.save( os.path.join( os.path.dirname(photometryFile), "photometry_data.npy"), photometryDataNumpy ); #Save the data as numpy file
#    photometryData = np.transpose(photometryDataNumpy);

photometrySheet = pd.read_csv( photometryFile, header=0, usecols=[0,3] ); #Load the data
#nonNanMask = photometrySheet["Analog In. | Ch.1 AIn-1 - Dem (AOut-2)_LowPass_dF/F0-Analog In. | Ch.1 AIn-1 - Dem (AOut-1)_LowPass_dF/F0"] > 0.; #Filtering NaN values (missing values)
#filteredPhotometrySheet = photometrySheet[nonNanMask]; #Filter the data
#photometryDataNumpy = filteredPhotometrySheet.to_numpy(); #Convert to numpy for speed
photometryDataNumpy = photometrySheet.to_numpy(); #Convert to numpy for speed
np.save( os.path.join( os.path.dirname(photometryFile), "photometry_data.npy"), photometryDataNumpy ); #Save the data as numpy file
photometryData = np.transpose(photometryDataNumpy);

plt.figure()
plt.plot(photometryData[0],photometryData[1])
    
#rawX = np.array([ np.mean( [  float(i) for i in photometryData[0][:][ int(n) : int(n+(samplingRateDoric*displayResolution)) ] ] )\
#                 for n in np.arange(0, len(photometryData[0][:]), (samplingRateDoric*displayResolution)) ]);
#    
#rawIsosbestic = np.array([ np.mean( [  float(i) for i in photometryData[1][:][ int(n) : int(n+(samplingRateDoric*displayResolution)) ] ] )\
#                          for n in np.arange(0, len(photometryData[1][:]), (samplingRateDoric*displayResolution)) ]);
#    
#rawCalcium = np.array([ np.mean( [  float(i) for i in photometryData[2][:][ int(n) : int(n+(samplingRateDoric*displayResolution)) ] ] )\
#                           for n in np.arange(0, len(photometryData[2][:]), (samplingRateDoric*displayResolution)) ]);

rawX = photometryData[0];
rawdF = photometryData[1];
    
X = rawX[int(startVideo*ratio):int(endVideo*ratio)];
X = X - X[0];
#
#Isosbestic = rawIsosbestic[startVideo:endVideo];
#Calcium = rawCalcium[startVideo:endVideo];
dF = rawdF[int(startVideo*ratio):int(endVideo*ratio)];

###############################################################################
#Reading the Video data
###############################################################################
print("\n");
print("Reading Video data");

videoClip = mpy.VideoFileClip(videoFile).subclip(t_start=startVideo, t_end=endVideo).crop(x1=807,y1=221,x2=1356,y2=485);

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
                
#                cottonGraph.set_data(x[0:i],y0[0:i]);
#                eventGraph.set_data(x[0:i],y1[0:i]);
#                calciumGraph.set_data(x[0:i],y2[0:i]);
                isosbesticGraph.set_data(x[0:i],y3[0:i]);
                
            except :
                
                print("Oups a problem occured")
                pass;
    
            last_frame = mplfig_to_npimage(liveFig);
            return last_frame;
        
        else :
            
            delta = i - thresh;
        
#            liveAx0.set_xlim(delta,i);
#            liveAx1.set_xlim(delta,i);
#            liveAx2.set_xlim(delta,i);
            liveAx3.set_xlim(delta,i);
            
            try :
                
#                cottonGraph.set_data(x[0:i],y0[0:i]);
#                eventGraph.set_data(x[0:i],y1[0:i]);
#                calciumGraph.set_data(x[0:i],y2[0:i]);
                isosbesticGraph.set_data(x[0:i],y3[0:i]);
                
            except :
                
                print("Oups a problem occured")
                pass;
    
            last_frame = mplfig_to_npimage(liveFig);
            return last_frame;
    
#    _FrameRate = vidClip.fps;
    _Duration = vidClip.duration;
    
    liveFig = plt.figure(figsize=(10,6), facecolor='white');
    
    gs = liveFig.add_gridspec(ncols=1, nrows=1);
    
#    #First live axis
#    liveAx0 = liveFig.add_subplot(gs[0, :]);
#    
#    liveAx0.set_title("Cotton height");
#    liveAx0.set_xlim([0, thresh]);
#    liveAx0.set_ylim([min(y0), max(y0)]);
#    
#    cottonGraph, = liveAx0.plot(x[0],y0[0],'-',color="blue",alpha=0.8,ms=1.);
#    
#    #Second live axis
#    liveAx1 = liveFig.add_subplot(gs[1, :]);
#    
#    liveAx1.set_title("Raster plot");
#    liveAx1.set_xlim([0, thresh]);
#    liveAx1.set_ylim([-0.1, 1.1]);
#    
#    eventGraph, = liveAx1.plot(x[0],y1[0],'-',color="blue",alpha=0.8,ms=1.);
    
#    #Third live axis
#    liveAx2 = liveFig.add_subplot(gs[2, :]);
#    
#    liveAx2.set_title("Calcium dependant trace (465nm) (mV)");
#    liveAx2.set_xlim([0, thresh]);
#    liveAx2.set_ylim([0.03, 0.08]);
#    
#    calciumGraph, = liveAx2.plot(x[0],y2[0],'-',color="red",alpha=0.8,ms=1.);
    
    #Fourth live axis
    liveAx3 = liveFig.add_subplot(gs[:, :]);
    
    liveAx3.set_title("Isosbestic trace (405nm) (mV)");
    liveAx3.set_xlim([0, thresh]);
    liveAx3.set_ylim([-5, 10]);

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
    
Cotton = []
Calcium = []
detectedPeaksMerged = []
    
LivePhotometryTrack(videoClip, X, Cotton, detectedPeaksMerged, Calcium, dF,\
                    displayThreshold, globalAcceleration=5, plotAcceleration=1./displayResolution);
        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
dbe = 3; #Distance Between Events
bl = 15; #Behavioral Bout Length;

def PhotometryMajorBoutDetection(events, distanceBetweenEvents, boutLength) :
    
    majorEventsPositions = [];
    majorEventsLengths = [];
    seeds = [];
    
    newBoutDetected = False;
    newMajorBoutDetected = False;
    
    posPotentialEvent = 0;
    
    distanceToNextEvent = 0;
    cumulativeEvents = 0;
    
    for pos, event in enumerate(events) :
        
        if event == True :
            
            cumulativeEvents += 1;
            
            if not newBoutDetected :
                
                newBoutDetected = True;
                posPotentialEvent = pos;
                seeds.append(pos);
                
            elif newBoutDetected :
                    
                distanceToNextEvent = 0;
                
                if cumulativeEvents >= boutLength :
                    
                    if not newMajorBoutDetected :
                        
                        newMajorBoutDetected = True; 
                        majorEventsPositions.append(posPotentialEvent);
                        
                    else :
                        
                        pass;
                    
                else :
                    
                    pass;

        elif event == False :
            
            if newBoutDetected :
            
                if distanceToNextEvent < distanceBetweenEvents :
                
                    distanceToNextEvent += 1;
            
                elif distanceToNextEvent >= distanceBetweenEvents :
                    
                    if newMajorBoutDetected :
                        
                        majorEventsLengths.append(cumulativeEvents);
                    
                    distanceToNextEvent = 0;
                    newBoutDetected = False;
                    newMajorBoutDetected = False; 
                    cumulativeEvents = 0;
    
    return majorEventsPositions, majorEventsLengths, seeds;
            
majorEventsPositions, majorEventsLengths, seeds = PhotometryMajorBoutDetection(detectedPeaksMerged, dbe, bl);

majorEvents = [True if n in majorEventsPositions else False for n, i in enumerate(detectedPeaksMerged)] 
seedsEvents = [True if n in seeds else False for n, i in enumerate(detectedPeaksMerged)]     

fig = plt.figure(figsize=(20,10));

ax0 = plt.subplot(111);
nesting, = ax0.plot(detectedPeaksMerged[7600:8500], color="blue", alpha=0.5);
seeds, = ax0.plot(seedsEvents[7600:8500], color="orange", alpha=1.);
major, = ax0.plot(majorEvents[7600:8500], color="red", alpha=1.);
ax0.set_xlabel("Time (s)");
ax0.set_ylabel("Nesting behavior bouts");
ax0.set_title("Nest-building raster plot over time");
ax0.legend(handles=[nesting, seeds, major], labels=["Nesting Bouts", "Start of nesting event",\
           "Start of major nesting event"], loc=2);
        
plt.savefig("/home/thomas.topilko/Desktop/Raster.png", dpi=200.);

plt.plot(detectedPeaksMerged, color="green")
plt.plot(seedsEvents, color="cyan")
plt.plot(majorEvents, color="blue")

dF = np.load(dFFile);
graphDistance = 10;
CalciumData = photometryData[2][:].tolist(); 

def ExtractCalciumDataWhenBehaving(posPeaks, calciumData, dist, samplingRateDoric, resample=1) :
    
    data = [];
    
    for p in posPeaks :  
        
        segment = calciumData[int((p*samplingRateDoric)-(dist*samplingRateDoric)) : \
                                int((p*samplingRateDoric)+((dist+1)*samplingRateDoric))];
                              
        resampled = [np.mean(segment[int(i) : int(i)+int(samplingRateDoric/resample)]) for i in np.arange(0, len(segment), int(samplingRateDoric/resample))]
        
        data.append(resampled);
        
#    data = np.array(data);
        
    return data;

from matplotlib.widgets import MultiCursor

zeroList = np.zeros((len(CalciumData)-len(dF)));
dFn = np.concatenate((zeroList, dF), axis=None)

fig = plt.figure()
ax0 = plt.subplot(211)
ax0.plot(CalciumData[0:12000*60*10]);
#ax0.plot(majorEvents[0:12000*60*10]);
ax1 = plt.subplot(212, sharex=ax0)
ax1.plot(dFn[0:12000*60*10]);
multi = MultiCursor(fig.canvas, (ax0, ax1), horizOn=True, vertOn=True, color='r', lw=1)
         
calciumDataAroundPeaks = ExtractCalciumDataWhenBehaving(majorEventsPositions, dFn,\
                                                        graphDistance, samplingRateDoric, resample=12000);

Mean = np.mean(calciumDataAroundPeaks, axis=0);
Std = np.std(calciumDataAroundPeaks, axis=0);
#plt.plot(Mean)

calciumDataAroundPeaksNew = [];
res = 12000
resPlot = int(12000/48)

for i in calciumDataAroundPeaks :
    
    sink = [];
    pos = np.arange(0, (2*res*10)+1, resPlot);
    
    for j in pos :
        print(j)
        mean = np.mean(i[j : j+resPlot])
        print(mean)
        sink.append(mean)
    calciumDataAroundPeaksNew.append(sink);
    
calciumDataAroundPeaksNew = np.array(calciumDataAroundPeaksNew)
MeanNew = np.mean(calciumDataAroundPeaksNew, axis=0);
plt.plot(MeanNew)

fig = plt.figure(figsize=(20,10));
ax0 = plt.subplot(211);
ax0.plot(np.arange(-graphDistance*resPlot, (graphDistance+1)*resPlot, 1), MeanNew, color="blue", alpha=0.5, lw=0.5);
ax0.set_xticks(np.arange(-graphDistance*resPlot,graphDistance*resPlot+1, resPlot*5));
ax0.set_xticklabels(np.arange(-10,10+1, 5));
ax0.set_xlim(-graphDistance*resPlot,graphDistance*resPlot+1);
ax0.set_ylim(min(Mean)+0.1*min(Mean),max(Mean)+0.1*max(Mean));
ax0.axvline(x=graphDistance, color='red', linestyle='--', lw=1);
ax0.set_xlabel("Time (s)");
ax0.set_ylabel("dF/F");
ax0.set_title("dF/F in function of time before & after nest-building initiation");
patch = patches.Rectangle((0,min(Mean)), width=np.mean(majorEventsLengths)*12000,\
                          height=(max(Mean)-min(Mean))*0.1, color='gray', lw=1, alpha=0.5);
ax0.add_patch(patch);

ax1 = plt.subplot(212);
heatmap = ax1.imshow(calciumDataAroundPeaksNew, cmap='viridis', interpolation='nearest', aspect="auto");
ax1.set_ylim(-0.5,3.5)
ax1.set_yticks(np.arange(0,4,1));
ax1.set_yticklabels(["Event {0}".format(i) for i in np.arange(0,len(calciumDataAroundPeaksNew))]);
ax1.set_xticks(np.arange(0,2*graphDistance*int(res/resPlot)+1,5*int(res/resPlot)));
ax1.set_xticklabels(np.arange(-10,10+1, 5));
ax1.axvline(x=graphDistance*int(res/resPlot), color='red', linestyle='--', lw=1);
ax1.set_xlabel("Time (s)");
ax1.set_title("dF/F for each individual event in function of time before & after nest-building initiation");
#cbar = plt.colorbar(heatmap, orientation="vertical");

plt.tight_layout();

plt.savefig("/home/thomas.topilko/Desktop/Photometry.png", dpi=200.)
#ax0.plot(np.arange(-boutDist, boutDist+1, 1), Mean+Std, color="blue", alpha=0.5);
#ax0.plot(np.arange(-boutDist, boutDist+1, 1), Mean-Std, color="blue", alpha=0.5);
#ax0.fill_between(np.arange(-boutDist, boutDist+1, 1), Mean, Mean+Std, color="blue", alpha=0.1);
#ax0.fill_between(np.arange(-boutDist, boutDist+1, 1), Mean, Mean-Std, color="blue", alpha=0.1);
