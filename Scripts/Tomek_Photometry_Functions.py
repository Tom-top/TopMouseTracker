#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:40:32 2019

@author: thomas.topilko
"""

import os;
import numpy as np;
import pandas as pd;
import datetime;
import matplotlib.pyplot as plt;
import matplotlib.patches as patches;
from matplotlib.widgets import MultiCursor;
import scipy.optimize as optimization;
from scipy.interpolate import UnivariateSpline, interp1d;
from scipy import signal;
import matplotlib;

def hoursMinutesSeconds(time) :
    
    delta = time%3600;
    h = (time - delta) / 3600;
    m = (delta - (delta%60)) / 60;
    s = (delta%60);
    
    return h, m, s;

def ConvertPhotometryData(csvFile) :
    
    npyFile = os.path.join( os.path.dirname(csvFile), "{0}.npy".format( os.path.basename(csvFile).split(".")[0] ) );
    
    if not os.path.exists(csvFile) :
        
        raise RuntimeError("{0} file doesn't exist !".format(csvFile));
    
    if not os.path.exists(npyFile) :
        
        print("\n");
        print("Converting CSV photometry data into NPY");
    
        photometrySheet = pd.read_csv( csvFile, header=1, usecols=np.arange(0,3) ); #Load the data
        nonNanMask = photometrySheet["AIn-1 - Dem (AOut-1)"] > 0.; #Filtering NaN values (missing values)
        
        filteredPhotometrySheet = photometrySheet[nonNanMask]; #Filter the data
        photometryDataNumpy = filteredPhotometrySheet.to_numpy(); #Convert to numpy for speed
        np.save(npyFile, photometryDataNumpy); #Save the data as numpy file
        
        print("Filetered : {0} points".format(len(photometrySheet)-len(filteredPhotometrySheet)));
        
        return npyFile;

    else :
        
        print("[WARNING] NPY file already existed, bypassing conversion step !");
        
        return npyFile;
    
def checkTimeShift(npy, SRD, plot=False, label=None) :
    
    SRD = int(SRD);
    photometryData = np.load(npy); #Load the NPY file
    
    #Cropping the data to the last second
    remaining = len(photometryData)%SRD;
    cropLength = int( len(photometryData) - remaining );
    
    cropped = photometryData[0:cropLength+1];
    clone = cropped.copy()
    
    expTime = [];
    fixedTime = [];
#    step = int(SRD/2)
    
    for n in np.arange(SRD, len(cropped), SRD-1) :  
        
        expTime.append(1 - (cropped[:,0][n-1] - cropped[:,0][n-SRD]));
#        dt = (1 - (cropped[:,0][n-1] - cropped[:,0][n-SRD])) / SRD;
#        clone[:,0][n-SRD : n-1] = clone[:,0][n-SRD : n-1]+dt;
#        fixedTime.append(clone[:,0][n-1] - clone[:,0][n-SRD]);
    
    expTime = np.array(expTime);
#    fixedTime = np.array(fixedTime);
    theoTime = np.arange(0, len(expTime), 1) ;
    
    if plot :
    
        fig = plt.figure(figsize=(5,3), dpi=200.);
        ax0 = plt.subplot(1,1,1);
        
        ax0.scatter(theoTime, expTime, s=0.1);
#        ax0.plot(np.cumsum(expTime));
        print(np.cumsum(expTime))
        
        minimum = min(expTime);
        maximum = max(expTime);
        r = maximum-minimum;
        print(minimum, maximum)
        
        ax0.set_ylim(minimum-r*0.1, maximum+r*0.1);
        
    ax0.set_title(label);
    
def movingAverage(arr, n=1) :
    
    ret = np.cumsum(arr, dtype=float);
    ret[n:] = ret[n:] - ret[:-n];
    newArr = ret[n - 1:] / n;
    newArr = np.insert(newArr, len(newArr), np.full(len(arr)-len(newArr), newArr[-1])) 
    
    return newArr;

def lowpassFilter(data, SRD, freq, order) :
    
    sampling_rate = SRD;
    nyq = sampling_rate/2;
    cutoff = freq;
    normalized_cutoff = cutoff/nyq;
    
    z, p, k = signal.butter(order, normalized_cutoff, output="zpk");
    lesos = signal.zpk2sos(z, p, k);
    filtered = signal.sosfilt(lesos, data);
    filtered = np.array(filtered);
    
    return filtered;

def interpolatedAndExtract(compressedX, finalX, source) :
    
    spl = interp1d(compressedX, source);
    sink = spl(finalX);
    
    return sink;

def plotIntermediatePhotometryPlots(data, **kwargs) :
    
    purpleLaser = "#8200c8"; #hex color of the calcium laser
    blueLaser = "#0092ff"; #hex color of the isosbestic laser
    
    if kwargs["function"] == None :
        
        sink = data;
        
    elif kwargs["function"] == "low_pass" :
        
        sink = [];
        
        for n, d in enumerate(data) :
            
            if n == 10 :
            
                sink.append(lowpassFilter(d, kwargs["SRD"], kwargs["freq"], kwargs["order"]));
                
            else :
                
                sink.append(d);
        
    elif kwargs["function"] == "moving_avg" :
        
        sink = [];
        
        for n, d in enumerate(data) :
            
            if n == 10 :
            
                sink.append(movingAverage(d, n=kwargs["roll_size"]));
                
            else :
                
                sink.append(d);
    
    plt.ion();
    
    fig = plt.figure(figsize=(10,5), dpi=200.);
    
    ax0 = plt.subplot(211);
    b, = ax0.plot(sink[0],sink[1],alpha=0.8,c=purpleLaser,lw=kwargs["lw"]);
    g, = ax0.plot(sink[0],sink[3],alpha=0.8,c="orange",lw=kwargs["lw"]);
    ax0.legend(handles=[b, g], labels=["isosbestic", "moving average"], loc=2, fontsize=kwargs["fsl"]);
    ax1 = plt.subplot(212);
    b, = ax1.plot(sink[0],sink[2],alpha=0.8,c=blueLaser,lw=kwargs["lw"]);
    g, = ax1.plot(sink[0],sink[5],alpha=0.8,c="orange",lw=kwargs["lw"]);
    ax1.legend(handles=[b, g], labels=["calcium", "moving average"], loc=2, fontsize=kwargs["fsl"]);
    ax0.set_title("Raw Isosbestic and Calcium signals", fontsize=kwargs["fst"]);
    ax1.set_xlabel("Time (s)", fontsize=kwargs["fsl"]);
    ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"]);
    ax1.tick_params(axis='both', which='major', labelsize=kwargs["fsl"]);
    
    if kwargs["save"] :
        
        plt.savefig(os.path.join(os.path.expanduser("~")+"/Desktop", "Isosbestic_Calcium_Raw.svg"), dpi=200.);

    fig = plt.figure(figsize=(10,5), dpi=200.);
    ax0 = plt.subplot(211);
    isos, = ax0.plot(sink[0],sink[4],alpha=0.8,c=purpleLaser,lw=kwargs["lw"]);
    ax1 = plt.subplot(212);
    calc, = ax1.plot(sink[0],sink[6],alpha=0.8,c=blueLaser,lw=kwargs["lw"]);
    ax0.legend(handles=[isos,calc], labels=["405 signal", "465 signal"], loc=2, fontsize=kwargs["fsl"]);
    ax0.set_title("Baseline Correction", fontsize=kwargs["fst"]);
    ax1.set_xlabel("Time (s)", fontsize=kwargs["fsl"]);
    ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"]);
    ax1.tick_params(axis='both', which='major', labelsize=kwargs["fsl"]);
    
    if kwargs["save"] :
        
        plt.savefig(os.path.join(os.path.expanduser("~")+"/Desktop", "Isosbestic_Calcium_Corrected.svg"), dpi=200.);

    fig = plt.figure(figsize=(10,5), dpi=200.);
    ax0 = plt.subplot(211);
    isos, = ax0.plot(sink[0],sink[7],alpha=0.8,c=purpleLaser,lw=kwargs["lw"]);
    ax1 = plt.subplot(212);
    calc, = ax1.plot(sink[0],sink[8],alpha=0.8,c=blueLaser,lw=kwargs["lw"]);
    ax0.legend(handles=[isos,calc], labels=["405 signal", "465 signal"], loc=2, fontsize=kwargs["fsl"]);
    ax0.set_title("Standardization", fontsize=kwargs["fst"]);
    ax1.set_xlabel("Time (s)", fontsize=kwargs["fsl"]);
    ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"]);
    ax1.tick_params(axis='both', which='major', labelsize=kwargs["fsl"]);
    
    if kwargs["save"] :
        
        plt.savefig(os.path.join(os.path.expanduser("~")+"/Desktop", "Isosbestic_Calcium_Standardized.svg"), dpi=200.);

#        fig = plt.figure(figsize=(10,5), dpi=200.);
#        ax = plt.subplot(111);
#        ax.scatter(movingAverage(standardizedIsosbestic, n=resample),movingAverage(standardizedCalcium, n=resample),alpha=0.5,s=1.);
#        ax.plot([min(movingAverage(standardizedIsosbestic, n=resample)),max(movingAverage(standardizedCalcium, n=resample))],\
#                 line, '--', c="red");
#        ax.set_xlabel("410 signal", fontsize=6.);
#        ax.set_ylabel("465 signal", fontsize=6.);
#        ax.set_title("Linear regression fit", fontsize=8.);
#        ax.tick_params(axis='both', which='major', labelsize=6);
#        
#        if save :
#            
#            plt.savefig(os.path.join(os.path.expanduser("~")+"/Desktop", "Isosbestic_Calcium_Fit.svg"), dpi=200.);
    
    fig = plt.figure(figsize=(10,5), dpi=200.);
    ax = plt.subplot(111);
    calc, = ax.plot(sink[0],sink[8],alpha=1.,lw=kwargs["lw"],c=blueLaser);
    isos, = ax.plot(sink[0],sink[9],alpha=1.,lw=kwargs["lw"],c=purpleLaser);
    ax.plot([sink[0][0], sink[0][-1]], [0,0], "--", color="red");
    ax.legend(handles=[isos,calc], labels=["fitted 405 signal", "465 signal"], loc=2, fontsize=kwargs["fsl"]);
    ax.set_xlabel("Time (s)", fontsize=kwargs["fsl"]);
    ax.set_title("Alignement of signals", fontsize=kwargs["fst"]);
    ax.tick_params(axis='both', which='major', labelsize=kwargs["fsl"]);
    
    if kwargs["save"] :
        
        plt.savefig(os.path.join(os.path.expanduser("~")+"/Desktop", "Isosbestic_Calcium_Aligned.svg"), dpi=200.);
    
    fig = plt.figure(figsize=(10,5), dpi=200.);
    ax = plt.subplot(111);
    df, = ax.plot(sink[0],sink[10],alpha=1.,lw=kwargs["lw"],c="green");
    ax.plot([sink[0][0], sink[0][-1]], [0,0], "--", color="red")
    ax.legend(handles=[df], labels=["dF/F"], loc=2, fontsize=kwargs["fsl"]);
    ax.set_xlabel("Time (s)", fontsize=kwargs["fsl"]);
    ax.set_title("dF/F", fontsize=kwargs["fst"]);
    ax.tick_params(axis='both', which='major', labelsize=kwargs["fsl"]);
    
    if kwargs["save"] :
        
        plt.savefig(os.path.join(os.path.expanduser("~")+"/Desktop", "Isosbestic_Calcium_dF.svg"), dpi=200.);

def LoadPhotometryData(npy, SRD, start, end, videoLength, rollAvg, optPlots=False, recompute=True, resample=1, plotargs=None) :
    
    SRD = int(SRD); #sampling rate of the Doric system
    plotargs["SRD"] = SRD;
    
    photometryData = np.load(npy); #Load the NPY file
    
    hi, mi, si = hoursMinutesSeconds(len(photometryData)/SRD);
    print("\n");
    print( "Theoretical length of measurement : " + str( datetime.timedelta(seconds=len(photometryData)/SRD) ) + " h:m:s" );
    print("Real length of measurement : " + str( datetime.timedelta(seconds=photometryData[:,0][-1]) ) + " h:m:s" );
    
    if photometryData[:,0][-1] - len(photometryData)/SRD != 0 :
        print("[WARNING] Shift in length of measurement : " + str( datetime.timedelta(seconds=(float(len(photometryData)/SRD) - float(photometryData[:,0][-1]))) ) + " h:m:s" );
#    print( (len(photometryData)/SRD)/60, (len(photometryData)/SRD)%60)

    #Cropping the data to the last second
    remaining = len(photometryData)%SRD;
    cropLength = int( len(photometryData) - remaining );
#    print( (len(photometryData[:cropLength])/SRD)/60, (len(photometryData[:cropLength])/SRD)%60)
    
    transposedPhotometryData = np.transpose( photometryData[:cropLength] ); #Transposed data
    
    print("\n");
    print( "Photometry recording length : " + str( datetime.timedelta(seconds=transposedPhotometryData.shape[1]/SRD) ) + " h:m:s" );
    
    rawX = np.array(transposedPhotometryData[0]); #time data
    compressedX = np.linspace(0, videoLength, len(rawX)); #compressed time data to fit the "real" time
    finalX = np.arange(start, end, 1/SRD); #time to be extracted
    
    rawIsosbestic = np.array(transposedPhotometryData[1]); #isosbestic data
    finalRawIsosbestic = interpolatedAndExtract(compressedX, finalX, rawIsosbestic); #data compressed in time
    
    rawCalcium = np.array(transposedPhotometryData[2]); #calcium data
    finalRawCalcium = interpolatedAndExtract(compressedX, finalX, rawCalcium); #data compressed in time

    #######################################################################
                            #Baseline correction#
    #######################################################################
    
    print("\n");
    print("Computing moving average for Isosbestic signal !");
                   
    funcIsosbestic = movingAverage(finalRawIsosbestic, n=SRD*rollAvg); #moving average for isosbestic data
    correctedIsosbestic = ( finalRawIsosbestic - funcIsosbestic ) / funcIsosbestic; #baseline correction for isosbestic
    
    print("\n");
    print("Computing moving average for Calcium signal !"); 

    funcCalcium = movingAverage(finalRawCalcium, n=SRD*rollAvg); #moving average for calcium data
    correctedCalcium = ( finalRawCalcium - funcCalcium ) / funcCalcium; #baseline correction for calcium
        
    #######################################################################
                            #Standardization#
    #######################################################################
    
    print("\n");
    print("Starting standardization for Isosbestic signal !");
    # (x-moy)/var
    
    standardizedIsosbestic = (correctedIsosbestic - np.median(correctedIsosbestic)) / np.std(correctedIsosbestic); #standardization for isosbestic
    standardizedCalcium = (correctedCalcium - np.median(correctedCalcium)) / np.std(correctedCalcium); #standardization for calcium
        
    #######################################################################
                        #Inter-channel regression#
    #######################################################################
    
    print("\n");
    print("Starting interchannel regression !");
    
    x1 = np.array([0.0, 0.0]); #init value
    
    def func2(x, a, b):
        
        return a + b*x #polynomial
    
    regrFit = optimization.curve_fit(func2, standardizedIsosbestic, standardizedCalcium, x1); #regression
    line = regrFit[0][0] + regrFit[0][1]*np.array([min(standardizedIsosbestic), max(standardizedIsosbestic)]); #regression line
         
    #######################################################################
                        #Signal alignement#
    #######################################################################
    
    print("\n");
    print("Starting signal alignement !");
    
    fittedIsosbestic = regrFit[0][0] + regrFit[0][1]*standardizedIsosbestic; #alignement of signals
        
    #######################################################################
                        #dF/F computation#
    #######################################################################
    
    print("\n");
    print("Computing dF/F(s) !");

    dF = standardizedCalcium - fittedIsosbestic; #computing dF/F
    
    #######################################################################
                        #For display#
    #######################################################################
    
    source = [finalX, finalRawIsosbestic, finalRawCalcium, funcIsosbestic, correctedIsosbestic,\
              funcCalcium, correctedCalcium, standardizedIsosbestic, standardizedCalcium,\
              fittedIsosbestic, dF];
    
    if optPlots :
        
        plotIntermediatePhotometryPlots(source, **plotargs);
    
    return finalX, fittedIsosbestic, standardizedCalcium, dF;
    
def LoadTrackingData(npy, fps, start, end) :
    
    rawCotton = np.load(npy); #Read the file

    Cotton = np.array([ np.mean( [  float(i) for i in rawCotton[ int(n) : int(n + fps) ] ] )\
                          for n in np.arange( 0, len(rawCotton), fps) ]);
    
    rawCotton = rawCotton[int(start*fps):int(end*fps)]
    Cotton = Cotton[start:end]; #Match the array length
    
    return rawCotton, Cotton;

def DetectRawPeaks(data, PAT, PTPD) :
    
    detectedPeaks = [];
    posDetectedPeaks = [];
    
    #Raw peack detection
    print("\n");
    print("Running peak detection algorithm");
    
    for i, dp in enumerate(data) : #For every data point
        
        if i <= (len(data) - 1) - PTPD : #If the point has enough points available in front of it
            
            if data[i]+PAT <= data[i+PTPD]\
            or data[i]-PAT >= data[i+PTPD] :
                
                detectedPeaks.append(True);
                posDetectedPeaks.append(i);
                
            else :
                
                detectedPeaks.append(False);
                
        else :
            
            detectedPeaks.append(False);
                
    print("Detected {0} peaks!".format(len(posDetectedPeaks)))
                
    return detectedPeaks, posDetectedPeaks;

def MergeClosePeaks(peaks, posPeaks, PMD) :
    
    #Merge close peaks
    print("\n");
    print("Running peak merging algorithm");
    
    detectedPeaksMerged = [];
    detectedPeaksMergedPositions = [];
    
    for pos, peak in enumerate(peaks) :
        
        d = pos;
        
        if peak :
                
            ind = posPeaks.index(pos);
            
            if ind < len(posPeaks) - 1 :
            
                if posPeaks[ind+1] - posPeaks[ind] < PMD :
                    
                    for i in np.arange(0, posPeaks[ind+1] - posPeaks[ind]) :
                        
                        detectedPeaksMerged.append(True);
                        detectedPeaksMergedPositions.append(d);
                        d+=1;
                        
                else :
                    
                    detectedPeaksMerged.append(True);
                    detectedPeaksMergedPositions.append(d);
                    d+=1;
            
            else :
                
                detectedPeaksMerged.append(True);
                detectedPeaksMergedPositions.append(d);
                d+=1;
                    
        else :
            
            if len(detectedPeaksMerged) <= pos :
            
                detectedPeaksMerged.append(False);
                
    return detectedPeaksMerged, detectedPeaksMergedPositions;

def DetectMajorBouts(events, DBE, BL, GD) :
    
    majorEventsPositions = [];
    majorEventsLengths = [];
    seedsPositions = [];
    
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
                seedsPositions.append(pos);
                
            elif newBoutDetected :
                    
                distanceToNextEvent = 0;
                
                if cumulativeEvents >= BL :
                    
                    if posPotentialEvent - GD > 0 and posPotentialEvent + GD < len(events) :
                    
                        if not newMajorBoutDetected :
                            
                            newMajorBoutDetected = True; 
                            majorEventsPositions.append(posPotentialEvent);
                            
                        else :
                            
                            pass;
                    
                else :
                    
                    pass;

        elif event == False :
            
            if newBoutDetected :
            
                if distanceToNextEvent < DBE :
                
                    distanceToNextEvent += 1;
            
                elif distanceToNextEvent >= DBE :
                    
                    if newMajorBoutDetected :
                        
                        majorEventsLengths.append(cumulativeEvents);
                    
                    distanceToNextEvent = 0;
                    newBoutDetected = False;
                    newMajorBoutDetected = False; 
                    cumulativeEvents = 0;
                    
    majorEvents = [True if n in majorEventsPositions else False for n, i in enumerate(events)];
    seedsEvents = [True if n in seedsPositions else False for n, i in enumerate(events)] ;
    
    return majorEvents, majorEventsPositions, majorEventsLengths, seedsEvents, seedsPositions;

def ExtractCalciumDataWhenBehaving(peaks, calcium, GD, SRD, resample=1) :
    
    data = [];
    
    for p in peaks :  
        
        segment = calcium[int((p*SRD)-(GD*SRD)) : \
                                int((p*SRD)+((GD+1)*SRD))];
        
        data.append(segment);
        
    return data;

def PeriEventPlot(data, length, resGraph, resHeatmap, GD, GDf, GDb, SRD, save=False, showStd=False, fileNameLabel="", cmap="viridis", lowpass=(2,2), norm=False) :
    
    
    resGraphData = [];
    resHeatmapData = [];
    resGraphFactor = SRD/resGraph;
    resHeatmapFactor = SRD/resHeatmap;
    
    for d in data :
        
        smoothedGraph = lowpassFilter(d, SRD, lowpass[0], lowpass[1]);
        
        resampledGraph = np.array([np.mean(smoothedGraph[int(i) : int(i)+int(resGraphFactor)]) for i in np.arange(0, len(smoothedGraph), int(resGraphFactor))]);
        resampledHeatmap = np.array([np.mean(smoothedGraph[int(i) : int(i)+int(resHeatmapFactor)]) for i in np.arange(0, len(smoothedGraph), int(resHeatmapFactor))]);
        resGraphData.append(resampledGraph);
        resHeatmapData.append(resampledHeatmap);
        
    resGraphData = np.array(resGraphData);
    resHeatmapData = np.array(resHeatmapData);

    meanDataAroundPeaks = np.mean(resGraphData, axis=0);
    stdDataAroundPeaks = np.std(resGraphData, axis=0);
    
    fig = plt.figure(figsize=(20,10));
    
    #First Plot
    ax0 = plt.subplot(2,1,1);
    
    if showStd :
        
        avg = ax0.plot(np.arange(0,len(meanDataAroundPeaks)), meanDataAroundPeaks, color="blue", alpha=0.5, lw=0.5);
    
        ax0.plot(np.arange(0,len(meanDataAroundPeaks)), meanDataAroundPeaks + stdDataAroundPeaks, color="blue", alpha=0.2, lw=0.5);
        ax0.plot(np.arange(0,len(meanDataAroundPeaks)), meanDataAroundPeaks - stdDataAroundPeaks, color="blue", alpha=0.2, lw=0.5);
        
        ax0.fill_between(np.arange(0,len(meanDataAroundPeaks)), meanDataAroundPeaks, meanDataAroundPeaks + stdDataAroundPeaks,\
                         color="blue", alpha=0.1);
        ax0.fill_between(np.arange(0,len(meanDataAroundPeaks)), meanDataAroundPeaks, meanDataAroundPeaks - stdDataAroundPeaks,\
                         color="blue", alpha=0.1);
                         
        ax0.set_ylim(min(meanDataAroundPeaks - stdDataAroundPeaks)+0.1*min(meanDataAroundPeaks - stdDataAroundPeaks),\
                     max(meanDataAroundPeaks + stdDataAroundPeaks)+0.1*max(meanDataAroundPeaks + stdDataAroundPeaks));
                     
        patch = patches.Rectangle((resGraph*GD,min(meanDataAroundPeaks - stdDataAroundPeaks)-0.1*min(meanDataAroundPeaks - stdDataAroundPeaks)),\
                              width=np.mean(length)*resGraph,\
                              height=(max(meanDataAroundPeaks + stdDataAroundPeaks)-min(meanDataAroundPeaks - stdDataAroundPeaks))*0.1,\
                              color='gray', lw=1, alpha=0.5);
        ax0.add_patch(patch);
                     
    else :
        
        avg = ax0.plot(np.arange(0,len(meanDataAroundPeaks)), meanDataAroundPeaks, color="blue", alpha=0.5, lw=2);
        
        ax0.set_ylim(min(meanDataAroundPeaks)+0.1*min(meanDataAroundPeaks),\
                     max(meanDataAroundPeaks)+0.1*max(meanDataAroundPeaks));
                     
        patch = patches.Rectangle((resGraph*GD,min(meanDataAroundPeaks)-0.1*min(meanDataAroundPeaks)),\
                              width=np.mean(length)*resGraph,\
                              height=(max(meanDataAroundPeaks)-min(meanDataAroundPeaks))*0.1,\
                              color='gray', lw=1, alpha=0.5);
        ax0.add_patch(patch);
    
    xRange = np.arange(0,len(meanDataAroundPeaks));
    ax0.plot((xRange[0], xRange[-1]), (0, 0), "--", color="blue", lw=1)

    ax0.set_xticks( np.arange((resGraph*GD)-(resGraph*GDb), (resGraph*GD)+(resGraph*GDf)+5*resHeatmap, 5*resHeatmap ) ); 
    ax0.set_xticklabels(np.arange(-GDb, GDf+5, 5));
    ax0.set_xlim((resGraph*GD)-(resGraph*GDb), (resGraph*GD)+(resGraph*GDf));
    vline = ax0.axvline(x=resGraph*GD, color='red', linestyle='--', lw=1);
    ax0.set_xlabel("Time (s)");
    ax0.set_ylabel("dF/F");
    ax0.set_title("dF/F in function of time before & after nest-building initiation");
    
    ax0.legend(handles=[vline, patch], labels=["Begining of bout","Average behavioral bout length"], loc=2);
    
    #Second Plot
    ax1 = plt.subplot(2,1,2);
    
    if not norm :
        ax1.imshow(resHeatmapData, cmap=cmap, aspect="auto"); 
    else :
        ax1.imshow(resHeatmapData, cmap=cmap, aspect="auto", norm=matplotlib.colors.LogNorm()); #norm=matplotlib.colors.LogNorm()
    
    ax1.set_ylim(-0.5, len(resHeatmapData)-0.5);
#    ax1.set_yticks(np.arange(0, len(resHeatmapData), 1));
        
    ax1.set_yticks([]);
        
    n = 0;
    
    for l in length :
        
        leftPlot = (resGraph*GD)-(resGraph*GDb);
        ax1.text(leftPlot-leftPlot*0.01, n, l, ha="center", va="center");
        n+=1
    
    ax1.set_xticks( np.arange((resGraph*GD)-(resGraph*GDb), (resGraph*GD)+(resGraph*GDf)+5*resHeatmap, 5*resHeatmap )  );
    ax1.set_xticklabels(np.arange(-GDb, GDf+5, 5));
    ax1.set_xlim((resGraph*GD)-(resGraph*GDb), (resGraph*GD)+(resGraph*GDf));
    ax1.axvline(x=resHeatmap*GD, color='red', linestyle='--', lw=1);
    ax1.set_xlabel("Time (s)");
    ax1.set_title("dF/F for each individual event in function of time before & after nest-building initiation");
    #cbar = plt.colorbar(heatmap, orientation="vertical");
    
    plt.tight_layout();
    
    if save :
        
        plt.savefig("/home/thomas.topilko/Desktop/Photometry{0}.svg".format(fileNameLabel), dpi=200.);
        
def PeakPlot(peaks, mergedPeaks, majorPeaks, seedPeaks, multiPlot=False, timeFrame=[], save=False) :
    
    fig = plt.figure(figsize=(20,10));
    fsl = 15.
    fst = 20.
    
    if timeFrame != [] : 
        peaks = np.array(peaks)[(timeFrame[0] < np.array(peaks)) &  (np.array(peaks) < timeFrame[1])];
        mergedPeaks = np.array(mergedPeaks)[(timeFrame[0] < np.array(mergedPeaks)) &  (np.array(mergedPeaks) < timeFrame[1])];
        seedPeaks = np.array(seedPeaks)[(timeFrame[0] < np.array(seedPeaks)) &  (np.array(seedPeaks) < timeFrame[1])];
        majorPeaks = np.array(majorPeaks)[(timeFrame[0] < np.array(majorPeaks)) &  (np.array(majorPeaks) < timeFrame[1])];
    
    if not multiPlot :
        
        ax0 = plt.subplot(1,1,1);
        ax0.plot(peaks, color="blue")
        ax0.plot(mergedPeaks, color="blue");
        ax0.plot(seedPeaks, color="orange");
        ax0.plot(majorPeaks, color="red");
        
    else :
        
        ax0 = plt.subplot(4,1,1);
        ax0.eventplot(peaks, color="blue", lineoffsets=0, linelengths=1);
        ax0.set_ylim(-0.5, 0.5);
        ax0.set_yticks([]);
        ax0.set_ylabel("Raw behavior bouts", rotation=0, va="center", ha="right", fontsize=fsl);
        
        ax0.set_title("Detection of major nesting events", fontsize=fst);
        ax0.tick_params(axis='both', which='major', labelsize=fsl);
        
        ax1 = plt.subplot(4,1,2, sharex=ax0, sharey=ax0);
        ax1.eventplot(mergedPeaks, color="blue", lineoffsets=0, linelengths=1);
        ax1.set_ylim(-0.5, 0.5);
        ax1.set_yticks([]);
        ax1.set_ylabel("Merged behavior bouts", rotation=0, va="center", ha="right", fontsize=fsl);
        ax1.tick_params(axis='both', which='major', labelsize=fsl);
        
        ax2 = plt.subplot(4,1,3, sharex=ax0, sharey=ax0);
        ax2.eventplot(seedPeaks, color="blue", lineoffsets=0, linelengths=1);
        ax2.set_ylim(-0.5, 0.5);
        ax2.set_yticks([]);
        ax2.set_ylabel("Start of each bout", rotation=0, va="center", ha="right", fontsize=fsl);
        ax2.tick_params(axis='both', which='major', labelsize=fsl);
        
        ax3 = plt.subplot(4,1,4, sharex=ax0, sharey=ax0);
        ax3.eventplot(majorPeaks, color="blue", lineoffsets=0, linelengths=1);
        ax3.set_ylim(-0.5, 0.5);
        ax3.set_yticks([]);
        ax3.set_ylabel("Major bouts", rotation=0, va="center", ha="right", fontsize=fsl);
        ax3.tick_params(axis='both', which='major', labelsize=fsl);
        
    if save :
        
        plt.savefig("/home/thomas.topilko/Desktop/PeakDetection.svg", dpi=200.);
        
def savePeriEventData(data, length, pos, sinkDir, mouse) :
    
    sink = np.array([data, length, pos]);
    
    np.save(os.path.join(sinkDir, "peData_{0}.npy".format(mouse)), sink);
    
def reorderByBoutSize(data, length, pos) :
    
    order = np.argsort(length);
    
    sinkLength = np.array(length)[order];
    sinkData = np.array(data)[order];
    sinkPos = np.array(pos)[order];
    
    return sinkData, sinkLength, sinkPos;

def filterShortBouts(data, length, pos, thresh) :
    
    sinkData = [];
    sinkLength = [];
    sinkPos = [];
    
    for d, l, p in zip(data, length, pos) :
        
        if l >= thresh :
            
            sinkData.append(d);
            sinkLength.append(l);
            sinkPos.append(p);
            
        else :
            
            pass;
            
    return sinkData, sinkLength, sinkPos;

def extractManualBehavior(raw, file) :

    peaks = np.full_like(raw, False);

    f = pd.read_excel(file, header=None);
    
    for s, e in zip(f.iloc[0][1:], f.iloc[1][1:]) :
        
        for i in np.arange(s, e, 1) :
            
            peaks[i] = True;
        
    return peaks;
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        