#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:47:34 2019

@author: thomas.topilko
"""

#print(Plot.totalTimeAutomatic)
#print(Plot.totalTimeManual)

#Plot.HeatMapPlot(bins=500,sigma=6);


#%%#######################################################################################################################################################
#Position Plot
##########################################################################################################################################################  

Plot = analysis.Plot(**trackerParameters);

Plot.TrackingPlot(1,plotB=False,plotA=True,cBefore='b',cAfter='r',limit=6);

#%%#######################################################################################################################################################
#Optimization
##########################################################################################################################################################  

import os;
import numpy as np;
import matplotlib as mpl;
import matplotlib.pyplot as plt;
from matplotlib.patches import Patch;

OptimizationDir = "/mnt/vol00-renier/Thomas/thomas.topilko/Inhibitory_Optimization_Raster"

minDist = 2; #1,3,5

Automatic = [];
Manual = [];

plt.figure(figsize=(10,7));
plt.title("PeakThresh_MinDist");

delta = 0.1;
alpha = 0.3;
_x = np.arange(0,2,0.1);
_y = np.arange(0,20,1.);

for folder in os.listdir(OptimizationDir) :
    
    path2Folder = os.path.join(OptimizationDir,folder)
    
    for file in os.listdir(path2Folder) :
        
        prefix = file.split(".")[0].split("_")[0]
        end = file.split(".")[0].split("_")[-1]
        
        if prefix == "PeakThresh" :
            
            if end == str(minDist) :

                Automatic.append(np.load(os.path.join(path2Folder,file)));
                                
        elif prefix == "Raster" :
            
            Manual.append(np.load(os.path.join(path2Folder,file)));
            
#Colors = [np.random.random(3) for i in range(len(Automatic))];  
Colors = ['green','blue','red','cyan']
Plots = [];

for i,j,k in zip(Automatic,Colors,Manual) :
    
    Z = [];
    
    for z in [x for sublist in i for x in sublist] :
    
        if z < k-k*delta :
                Z.append(np.nan)
                
        elif z > k+k*delta :
                Z.append(np.nan)
                
        else :
            Z.append(1)
            
    Z = np.array(Z).reshape(((len(_x),len(_y))))
    
    plot = plt.imshow(Z, cmap = mpl.colors.ListedColormap([j]),alpha=alpha, vmin=0, vmax=1)
    Plots.append(plot)
    
legend_elements = [Patch(facecolor= Colors[0],label='data 1',alpha=alpha),
                   Patch(facecolor= Colors[1],label='data 2',alpha=alpha),
                   Patch(facecolor= Colors[2],label='data 3',alpha=alpha),
                   Patch(facecolor= Colors[3],label='data 4',alpha=alpha),]

plt.legend(handles=legend_elements, loc='upper left');
plt.xlabel(r'$\alpha$', fontsize = 12); #"Threshold parameter for peak detection"
plt.ylabel(r'$\beta$', fontsize = 12); #"Distance parameter between behavior bouts"
plt.yticks(np.arange(len(_y)),[round(a,2) for a in _y]);
plt.xticks(np.arange(len(_x)),[round(a,2) for a in _x]);
plt.tick_params(axis='x', rotation=90,labelsize=10);
plt.tick_params(axis='y', labelsize=10);
plt.grid(linestyle='--', linewidth=0.5);
plt.gca().invert_yaxis();
plt.tight_layout();
plt.show();

plt.savefig(os.path.join("/mnt/raid/TopMouseTracker/Optimization","Optimization_Plot_10p_{0}.png").format(minDist))
        
#%%############################################################################
#Future
###############################################################################  

peakDist = 2;

Plot = analysis.Plot(**trackerParameters);
Plot.DataManual();

X = [];
Y = [];
Z = [];

_x = np.arange(0,2,0.1);
_y = np.arange(0,20,1.);
pos = 0

for y in _y :
    
    pos+=1
    
    print(pos/len(_y))
    
    for x in _x :
    
        z = Plot.OptimizeSegmentation([x,y],peakDist)
        if z[0] > Plot.totalTimeManual+50 :
            z[0] = 0;
        Z.append(z[0]);
        X.append(x);
        Y.append(y)

#peakThresh = 0, minDist = 1
Z = np.array(Z).reshape(((len(_x),len(_y))))

np.save(os.path.join(Plot._args["main"]["resultDir"],"PeakThresh_MinDist_Mouse_{0}_0-2_0-20_{1}.npy".format(mainParameters["mouse"],peakDist)
