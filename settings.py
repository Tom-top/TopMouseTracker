#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:51:08 2019

@author: Thomas TOPILKO
"""

import os
import sys

# Checking the operating system
# platforms: {
#  linux: "linux" or "linux2"
#  os_x: "darwin"
#  windows: "win32"}
_platform = sys.platform

def TopMouseTrackerPath():
    
    '''Returns path to the TopMouseTracker software
    
    Returns:
        str: path to TopMouseTracker
    '''
    
    fn = os.path.split(__file__);
    fn = os.path.abspath(fn[0]);
    return fn;

_topMouseTrackerPath = TopMouseTrackerPath();

_mainPath = os.path.expanduser("~");
_desktopPath = _mainPath+"/Desktop";

screenResString = os.popen("xrandr | grep '*'").read();

if not screenResString == '':
    resolution = screenResString.split()[0];
    _width, _height = resolution.split('x');
    _width = int(_width);
    _height = int(_height);