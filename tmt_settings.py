#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:51:08 2019

@author: Thomas TOPILKO
"""

import os
import sys

# Checking the operating system

# Linux : platform = "linux" or "linux2"
# OS X : platform = "darwin"
# Windows : platform = "win32"

_platform = sys.platform

_root_path = os.path.expanduser("~")
_desktop_path = _root_path + "/Desktop"

screen_res_str = os.popen("xrandr | grep '*'").read()

if not screen_res_str == '':
    resolution = screen_res_str.split()[0]
    _width, _height = resolution.split('x')
    _width = int(_width)
    _height = int(_height)
