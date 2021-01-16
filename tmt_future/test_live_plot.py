#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:13:05 2019

@author: thomas.topilko
"""

import os

import numpy as np

from natsort import natsorted
import matplotlib.pyplot as plt

import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoFileClip, VideoClip, clips_array, ImageSequenceClip


path = "/mnt/raid/TopMouseTracker/190305-01/255"
save = "/home/thomas.topilko/Desktop"

file = "/mnt/raid/TopMouseTracker/190305-01/255/Data_255_CottonPixelIntensities.npy"
rasterFile = "/mnt/raid/TopMouseTracker/190305-01/255/Data_255_Raster.npy"
refPtFile = "/mnt/raid/TopMouseTracker/190305-01/255/Data_255_refPt.npy"

refPt = np.load(refPtFile)

fps = 16.9

tStart = int((0)*fps)
tEnd = int((22*60+0)*fps)
#tEnd = (1*3600+25*60+19)*fps

#duration= 29*fps
duration= abs(tEnd-tStart)

fileList = ["/mnt/raid/TopMouseTracker/190305-01/255/segmentation/"+x for x in natsorted(os.listdir("/mnt/raid/TopMouseTracker/190305-01/255/segmentation"))]
#fileList = [x for x in fileList[::fps]]

#video = mpy.VideoFileClip("/mnt/raid/TopMouseTracker/190220-01/254/Tracking_254_0.avi",audio=False)#.subclip(tStart/fps,tEnd/fps)
video = mpy.ImageSequenceClip(fileList,fps=1)#.subclip(tStart/fps,tEnd/fps)
#video = video.crop(x1 = refPt[0][0], y1 = refPt[0][1], x2 = refPt[1][0], y2 = refPt[1][1])

thresh = int(10*fps)


fig, (ax0, ax1) = plt.subplots(2,figsize=(5,3), facecolor='white')
x = np.arange(tStart,tEnd)
y = np.load(file)
y = [np.mean(y[int(i):int(i+fps)]) for i in np.arange(0,len(y),fps)]
z = np.load(rasterFile)
ax0.set_title("Cotton height over time")
ax0.set_ylim(110,130)
ax0.set_xlim(tStart+1,tStart+thresh)
ax1.set_title("Raster over time")
ax1.set_ylim(-0.5,1.5)
ax1.set_xlim(tStart+1,tStart+thresh)

#array = np.concatenate((y[0:1],np.zeros(thresh-1)))
line, = ax0.plot(x[0:1],y[0:1],lw=2,c="blue",alpha=0.5)
patch, = ax1.plot(x[0:1],z[0:1],drawstyle="steps",lw=2,color="blue",alpha=0.5)

plt.tight_layout()

# ANIMATE WITH MOVIEPY (UPDATE THE CURVE FOR EACH t). MAKE A GIF.

def make_frame_mpl(t):
    global thresh,ax0,ax1
    
    i = int(t)
    if i < thresh :
        line.set_data(x[0:i],y[tStart:tStart+i])
        patch.set_data(x[0:i],z[tStart:tStart+i])
        #ax1.fill_between(x[i-1:i], 0, z[i-1:i],color="blue",alpha=0.5)
#        for X, p in zip(x[0:i],patch) :
#            if X == 1:
#                p.set_height(1);
        last_frame = mplfig_to_npimage(fig)
        return last_frame
    else :
        delta = i - thresh
        
        ax0.set_xlim(tStart+delta,tStart+i)
        ax1.set_xlim(tStart+delta,tStart+i)
        line.set_data(x[delta:i],y[tStart+delta:tStart+i])
        patch.set_data(x[delta:i],z[tStart+delta:tStart+i])
        #ax1.fill_between(x[i-1:i], 0, z[i-1:i],color="blue",alpha=0.5)
#        for X, p in zip(x[delta:i],patch) :
#            if X == 1:
#                p.set_height(1)
        last_frame = mplfig_to_npimage(fig)
        return last_frame
    
#def raster_mpl(t):
#    global thresh,ax1
#    
#    i = int(t)
#    if i < thresh :
#        patch.set_data(x[0:i],z[tStart:tStart+i])
#        last_frame = mplfig_to_npimage(fig)
#        return last_frame
#    else :
#        delta = i - thresh
#        
#        ax1.set_xlim(tStart+delta,tStart+i)
#        patch.set_data(x[delta:i],z[tStart+delta:tStart+i])
#        last_frame = mplfig_to_npimage(fig)
#        return last_frame

animation = mpy.VideoClip(make_frame_mpl, duration=len(y))
#animation2 = mpy.VideoClip(raster_mpl, duration=len(z))
#animation.speedx(fps).write_gif(os.path.join(save,"sinc_mpl.gif"), fps=10)

final_clip = clips_array([[clip.margin(2, color=[255,255,255]) for clip in
                [(video.resize(0.5)).speedx(fps), animation.speedx(1)]]],
                bg_color=[255,255,255])

#final_clip.write_gif(os.path.join(save,'test.gif'), fps=1.)
final_clip.write_videofile(os.path.join(save,'test3.mp4'), fps=10)
