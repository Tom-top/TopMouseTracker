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

dataFile = "/home/thomas.topilko/Desktop/Female_808/Trace_1.csv";
videoFile = "/home/thomas.topilko/Desktop/Female_808/19-11-2019_11-42-23/Raw_Video_Mice_808_19-11-2019_11-42-23.avi";

rawSheet = pd.read_csv(dataFile, header=1);
sheet = pd.read_csv(dataFile, header=1, usecols=np.arange(0,3)); #usecols=np.arange(0,9)

nonNanMask = sheet["AIn-1 - Dem (AOut-1)"] > 0.;
sheetNew = sheet[nonNanMask];
#delayToStart = sheet["Time(s)"][nonNanMask].iloc[0];
numpyArray = sheetNew.to_numpy();
np.save( os.path.join("/home/thomas.topilko/Desktop","data.npy"), numpyArray );

data = np.transpose(numpyArray);

samplingRate = 12000;
resolution = 1;

X = np.array([ np.mean( [  float(i) for i in data[0][:][ int(n) : int(n+(samplingRate*resolution)) ] ] ) for n in np.arange(0, len(data[0][:]), (samplingRate*resolution)) ]);
Y0 = np.array([ np.mean( [  float(i) for i in data[1][:][ int(n) : int(n+(samplingRate*resolution)) ] ] ) for n in np.arange(0, len(data[1][:]), (samplingRate*resolution)) ]);
Y1 = np.array([ np.mean( [  float(i) for i in data[2][:][ int(n) : int(n+(samplingRate*resolution)) ] ] ) for n in np.arange(0, len(data[2][:]), (samplingRate*resolution)) ]);

start = 10*60;
end = 40*60;

#dataDuration = 200;
Xnew = X[start:end];
Xnew = Xnew - Xnew[0];
Y0new = Y0[start:end];
Y1new = Y1[start:end];

#vidClip = mpy.VideoFileClip(videoFile).subclip(t_start=0, t_end=dataDuration);
#testFrame = vidClip.get_frame(100);
#plt.imshow(testFrame);

#vidClip = mpy.VideoFileClip(videoFile).subclip(t_start=start, t_end=end);
vidClip = mpy.VideoFileClip(videoFile).subclip(t_start=start, t_end=end).crop(x1=870,y1=494,x2=1401,y2=741);
#vidClip = mpy.VideoFileClip(videoFile).subclip(t_start=0, t_end=dataDuration).crop(x1=931,y1=153,x2=1475,y2=410);
threshDisplay = int(50*vidClip.fps);

def LivePhotometryTrack(vidClip, x, y0, y1, thresh, acceleration=1, showHeight=True, showTrace=True) :
    
    def make_frame_mpl(t):
         
        i = int(t);
        
        if i < thresh :
            
            try :
                
#                print("\n")
#                print(isosbesticGraph)
                isosbesticGraph.set_data(x[0:i],y0[0:i]);
                calciumGraph.set_data(x[0:i],y1[0:i]);
#                trajectoryGraph.set_data(list(zip(*Y[0:i]))[0],list(zip(*Y[0:i]))[1]);
                
            except :
                
                print("Oups a problem occured")
                pass;
    
            last_frame = mplfig_to_npimage(liveFig);
            return last_frame;
        
        else :
            
            delta = i - thresh;
        
            liveAx0.set_xlim(delta,i);
            liveAx1.set_xlim(delta,i);
            
            try :
            
                isosbesticGraph.set_data(x[0:i],y0[0:i]);
                calciumGraph.set_data(x[0:i],y1[0:i]);
#                trajectoryGraph.set_data(list(zip(*Y[0:i]))[0],list(zip(*Y[0:i]))[1]);
                
            except :
                
                print("Oups a problem occured")
                pass;
    
            last_frame = mplfig_to_npimage(liveFig);
            return last_frame;
    
#    _FrameRate = vidClip.fps;
    _Duration = vidClip.duration;
    
    liveFig = plt.figure(figsize=(5,3), facecolor='white');
    
    gs = liveFig.add_gridspec(ncols=1, nrows=2);
    
    liveAx0 = liveFig.add_subplot(gs[1, :]);
    
    liveAx0.set_title("Isosbestic trace (405nm) (mV)");
    liveAx0.set_xlim([0, thresh]);
#    liveAx0.set_ylim([min(y0), max(y0)]);
    liveAx0.set_ylim([0.03, 0.08]);
#    liveAx2.set_aspect("equal");

    isosbesticGraph, = liveAx0.plot(x[0],y0[0],'-o',color="green",alpha=0.8,ms=1.);
    
    liveAx1 = liveFig.add_subplot(gs[0, :]);
    
    liveAx1.set_title("Calcium dependant trace (465nm) (mV)");
    liveAx1.set_xlim([0, thresh]);
#    liveAx1.set_ylim([min(y1), max(y1)]);
    liveAx1.set_ylim([0.03, 0.08]);
    
    calciumGraph, = liveAx1.plot(x[0],y1[0],'-o',color="red",alpha=0.8,ms=1.);
    
#    plt.gca().invert_yaxis();
#        plt.gca().invert_xaxis();
    plt.tight_layout();
    
    _acceleration = acceleration;
    
#    a = make_frame_mpl(1000);
#    return a;
    
    anim = mpy.VideoClip(make_frame_mpl, duration=(_Duration*1));
    
    finalClip = clips_array([[clip.margin(2, color=[255,255,255]) for clip in
                    [(vidClip.speedx(_acceleration)), anim.speedx(_acceleration)]]],
                    bg_color=[255,255,255]); #.speedx(self._FrameRate)
    
    finalClip.write_videofile(os.path.join("/home/thomas.topilko/Desktop",'PhotoMetry_Tracking.mp4'), fps=10);
    
LivePhotometryTrack(vidClip, Xnew, Y0new, Y1new, threshDisplay, acceleration=5);
























        
