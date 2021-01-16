#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:51:07 2020

@author: thomas.topilko
"""

import os
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import xlwt

time_cutoff = 5*3600 #seconds
jitter_filter = 0.6
artefact_filter = 5
wb = xlwt.Workbook()
ws = wb.add_sheet('sheet')
ws.write(0, 0, "name")
ws.write(0, 1, "distance_traveled")

metadata_file = "/home/thomas.topilko/Desktop/Tracking/Metadata_Analysis.xlsx"

def compute_distance_traveled(file) :
        
    d = np.load(file)
    distance = [ sqrt( (d[i+1][0]-d[i][0])**2 + (d[i+1][1]-d[i][1])**2 ) for i in np.arange(0, len(d)-1, 1) ]
    
    return np.array(distance)

for n, a in enumerate(["183", "183bis", "193", "193bis", "194", "194bis", "195", "195bis",\
                       "196", "196bis", "204", "204bis", "255", "255bis", "260", "260bis",\
                       "262", "262bis", "263", "263bis", "238", "238bis", "254", "254bis",\
                       "253", "253bis", "256", "256bis", "263", "263bis", "252", "252bis",\
                       "261", "261bis"]) :
    
    dist_file = "/home/thomas.topilko/Desktop/Tracking/Tracking_Results/Experiment/Mouse_Data_All_{0}_Distances.npy".format(a)
    pos_file = "/home/thomas.topilko/Desktop/Tracking/Tracking_Results/Experiment/Mouse_Data_All_{0}_Points.npy".format(a)
    refPt_file = "/home/thomas.topilko/Desktop/Tracking/Tracking_Results/Experiment/Mouse_Data_All_{0}_refPt.npy".format(a)
    
    metadata = pd.read_excel(metadata_file)
    
    if os.path.exists(dist_file) :
        
        try :
            animal = int(dist_file.split("/")[-1].split("_")[-2])
        except :
            animal = str(dist_file.split("/")[-1].split("_")[-2])
        
        metadata_animal = metadata[metadata["name"] == animal]
        cage_width = float(metadata_animal["cage_width"]) #cm
        cage_length = float(metadata_animal["cage_length"]) #cm
        framerate = float(metadata_animal["framerate"])
        
        distance = np.load(dist_file)
        real_distance = distance[: int(time_cutoff*framerate)]
        real_distance[real_distance < jitter_filter] = 0

    else :
        
        try :
            animal = int(pos_file.split("/")[-1].split("_")[-2])
        except :
            animal = str(pos_file.split("/")[-1].split("_")[-2])
            
        metadata_animal = metadata[metadata["name"] == animal] 
        cage_width = float(metadata_animal["cage_width"]) #cm
        cage_length = float(metadata_animal["cage_length"]) #cm
        framerate = float(metadata_animal["framerate"])
        
        refPt = np.load(refPt_file)
        pixels_per_cm = np.mean( [((refPt[1][0] - refPt[0][0]) / cage_length), ((refPt[1][1] - refPt[0][1]) / cage_width)] )
        _positions = np.load(pos_file)
        distance = compute_distance_traveled(pos_file)
        
        print("Total time for mouse {0} : {1}s".format(animal, len(distance)/framerate))
        
        crop_distance = distance[: int(time_cutoff*framerate)]
        real_distance = distance/pixels_per_cm
        real_distance[real_distance < jitter_filter] = 0
        real_distance[real_distance > artefact_filter] = 0
    
#    if a in ["183bis", "195bis"] :
#        fig = plt.figure(figsize=(10,10))
#        ax0 = plt.subplot(2,1,2)
#        ax0.plot(real_distance)
#        ax1 = plt.subplot(2,1,1)
#        ax1.plot(_positions[:, 0], _positions[:, 1], "-o", alpha=1, markersize=2)
#        ax1.set_title("Mouse {0}".format(a))
#    
    dist_traveled = np.sum(real_distance)
    print("Distance traveled by mouse {0} : {1}m".format(animal, dist_traveled/100))
    ws.write(n+1, 0, animal)
    ws.write(n+1, 1, dist_traveled/100)

wb.save(os.path.join("/home/thomas.topilko/Desktop", "Distance_Traveled.xls"))

#%%

time_cutoff = 1*3600 #seconds
jitter_filter = 0.6
artefact_filter = 5
wb = xlwt.Workbook()
ws = wb.add_sheet('sheet')
ws.write(0, 0, "name")
ws.write(0, 1, "distance_traveled")

metadata_file = "/home/thomas.topilko/Desktop/Tracking/Metadata_Analysis.xlsx"

def compute_distance_traveled(file) :
        
    d = np.load(file)
    distance = [ sqrt( (d[i+1][0]-d[i][0])**2 + (d[i+1][1]-d[i][1])**2 ) for i in np.arange(0, len(d)-1, 1) ]
    
    return np.array(distance)

for n, a in enumerate(["68", "69", "70", "72", "74", "77", "78", "75", "82", "76",\
                       ]) :
    
    dist_file = "/home/thomas.topilko/Desktop/Tracking/Tracking_Results/Experiment/Data_{0}_Distances.npy".format(a)
    pos_file = "/home/thomas.topilko/Desktop/Tracking/Tracking_Results/Experiment/Data_{0}_Points.npy".format(a)
    refPt_file = "/home/thomas.topilko/Desktop/Tracking/Tracking_Results/Experiment/Data_{0}_refPt.npy".format(a)
    
    metadata = pd.read_excel(metadata_file)
    
    if os.path.exists(dist_file) :
        
        try :
            animal = int(dist_file.split("/")[-1].split("_")[-2])
        except :
            animal = str(dist_file.split("/")[-1].split("_")[-2])
        
        metadata_animal = metadata[metadata["name"] == animal]
        cage_width = float(metadata_animal["cage_width"]) #cm
        cage_length = float(metadata_animal["cage_length"]) #cm
        framerate = float(metadata_animal["framerate"])
        
        distance = np.load(dist_file)
        real_distance = distance[: int(time_cutoff*framerate)]
        real_distance[real_distance < jitter_filter] = 0

    else :
        
        try :
            animal = int(pos_file.split("/")[-1].split("_")[-2])
        except :
            animal = str(pos_file.split("/")[-1].split("_")[-2])
            
        metadata_animal = metadata[metadata["name"] == animal] 
        cage_width = float(metadata_animal["cage_width"]) #cm
        cage_length = float(metadata_animal["cage_length"]) #cm
        framerate = float(metadata_animal["framerate"])
        
        refPt = np.load(refPt_file)
        pixels_per_cm = np.mean( [((refPt[1][0] - refPt[0][0]) / cage_length), ((refPt[1][1] - refPt[0][1]) / cage_width)] )
        print(pixels_per_cm)
        break
        _positions = np.load(pos_file)
        distance = compute_distance_traveled(pos_file)
        
        print("Total time for mouse {0} : {1}s".format(animal, len(distance)/framerate))
        
        crop_distance = distance[: int(time_cutoff*framerate)]
        real_distance = distance/pixels_per_cm
        real_distance[real_distance < jitter_filter] = 0
        real_distance[real_distance > artefact_filter] = 0
    
    if a in ["183bis", "195bis"] :
        fig = plt.figure(figsize=(10,10))
        ax0 = plt.subplot(2,1,2)
        ax0.plot(real_distance)
        ax1 = plt.subplot(2,1,1)
        ax1.plot(_positions[:, 0], _positions[:, 1], "-o", alpha=1, markersize=2)
        ax1.set_title("Mouse {0}".format(a))
    
    dist_traveled = np.sum(real_distance)
    print("Distance traveled by mouse {0} : {1}m".format(animal, dist_traveled/100))
    ws.write(n+1, 0, animal)
    ws.write(n+1, 1, dist_traveled/100)

wb.save(os.path.join("/home/thomas.topilko/Desktop", "Distance_Traveled_2.xls"))


#%%

working_dir = "/network/lustre/dtlake01/renier/Thomas/thomas.topilko/Experiments/Nesting_Project/Behavior/Slices_cFOS_Ucn1/Distance_Traveled"
time_cutoff = 1*3600
cage_width = 22
cage_length = 36
jitter_filter = 0.6
n = 0
wb = xlwt.Workbook()
ws = wb.add_sheet('sheet')
ws.write(0, 0, "name")
ws.write(0, 1, "distance_traveled")

for f in os.listdir(working_dir) :
    
    path = os.path.join(working_dir, f)
    if f.split(".")[-1] == "csv" :
        
        animal = f.split(".")[0].split("_")[-1]
        
        raw = pd.read_csv(path, sep='delimiter', header=None)
        print(raw.iloc[0][0])
        if not animal == "15" :
            refPt = [eval(raw.iloc[0][0].split('"')[1]), eval(raw.iloc[0][0].split('"')[3])]
        else :
            refPt = [eval(raw.iloc[0][0].split('"')[1]), eval(raw.iloc[0][0].split('"')[5])]
        pixels_per_cm = np.mean( [((refPt[1][0] - refPt[0][0]) / cage_length), ((refPt[1][1] - refPt[0][1]) / cage_width)] )
        
        _positions = []
        for p in raw.iloc[-1][0].split('"') :
            try :
                _positions.append(eval(p))
            except :
                pass
            
        distance = [ sqrt( (_positions[i+1][0]-_positions[i][0])**2 + (_positions[i+1][1]-_positions[i][1])**2 ) for i in np.arange(0, len(_positions)-1, 1) ]
        
        crop_distance = distance[len(distance)-time_cutoff :]
        real_distance = crop_distance/pixels_per_cm
        real_distance[real_distance < jitter_filter] = 0
        real_distance[real_distance > artefact_filter] = 0
        
        dist_traveled = np.sum(real_distance)
        print("Distance traveled by mouse {0} : {1}m".format(animal, dist_traveled/100))
        ws.write(n+1, 0, animal)
        ws.write(n+1, 1, dist_traveled/100)
        n+=1
    
wb.save(os.path.join("/home/thomas.topilko/Desktop", "Distance_Traveled_3.xls"))













