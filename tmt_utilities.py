#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:56:59 2019

@author: Thomas TOPILKO
"""

import os
import shutil

colors = dict(white="\033[1;37m", yellow="\033[1;33m", green="\033[1;32m", blue="\033[1;34m", cyan="\033[1;36m",
              red="\033[1;31m", magenta="\033[1;35m", black="\033[1;30m", darkwhite="\033[0;37m",
              darkyellow="\033[0;33m", darkgreen="\033[0;32m", darkblue="\033[0;34m", darkcyan="\033[0;36m",
              darkred="\033[0;31m", darkmagenta="\033[0;35m", darkblack="\033[0;30m", bold="\033[1m", off="\033[0;0m")


def print_color_message(msg, color):
    """Prints a colored message in the console

    Params :
        msg (str) : the message to be printed
        color (str) : the color to be used for the message (see format in params.colors)

    Returns :
        prints the colored message in the console
    """

    if isinstance(msg, str):
        if color in colors:
            print(colors[color] + msg + colors["off"])
        else:
            print("\n")
            print_color_message("[WARNING] The color {0} is not available!".format(color), "darkred")
    else:
        print("\n")
        print_color_message("[WARNING] The message {0} is not in the correct format!".format(msg), "darkred")


def print_loading_bar(percent):
    """Prints a loading bar in the console
       e.g (0%) = '..................................................................................................................'
           (11.5%) = '######............................................................................................................'
           (100%) = '##################################################################################################################'

    Params :
        percent (int/float) : the % of the ongoing operation (e.g : 11.5 = 11.5%; 100 = 100%)

    Returns :
        prints the loading bar in the console
    """

    if percent >= 0:
        pattern = ['.'] * 100
        for n, i in enumerate(pattern):
            if n < percent:
                pattern[n] = '#'
            else:
                pattern[n] = '.'
        pattern = "".join(pattern)
        return pattern
    else:
        print_color_message("[WARNING] A progress cannot be negative right?", "darkred")


def get_h_m_s(time):
    """Function that takes a time variable (s) and converts it into (hh,mm,ss) format

    Params :
        time (int) : the number of seconds of a time/event

    Returns :
        returns the time in (hh,mm,ss) format
    """

    if time >= 0:
        remaining_minutes = time % 3600
        remaining_seconds = remaining_minutes % 60
        hours = int(time / 3600)
        minutes = int(remaining_minutes / 60)
        seconds = int(remaining_seconds)
        return hours, minutes, seconds
    else:
        print_color_message("[WARNING] Time cannot be negative right?", "darkred")


def check_and_create_dir(directory):
    """ Function that checks is a certain directory exists. If not : creates it

    Params :
        directory (str) : path to the directory to be created

    Output :
        Creates the designed directory
    """

    if isinstance(directory, str):
        if os.path.dirname(directory) != "":
            if not os.path.exists(directory):
                os.mkdir(directory)
                print_color_message("[INFO] {0} has been created".format(directory), "darkgreen")
            else:
                pass
        else:
            raise RuntimeError("{0} is not a valid directory!".format(directory))
    else:
        raise RuntimeError("{0} is not a valid directory!".format(directory))


def clear_directory(directory):
    if isinstance(directory, str):
        if os.path.dirname(directory) != "":
            if os.listdir(directory) != []:
                Input = input("Type y/[n] to clean {0} :   ".format(directory))
                if Input == "y":
                    for obj in os.listdir(directory):
                        try:
                            os.remove(os.path.join(directory, obj))
                        except:
                            shutil.rmtree(os.path.join(directory, obj))
                elif Input == "n" or Input == "":
                    print_color_message("{0} was kept untouched!".format(directory), "darkgreen")
                else:
                    print_color_message("Wrong input, {0} was kept untouched!".format(directory), "darkgreen")
            else:
                print_color_message("{0} was already empty!".format(directory), "darkgreen")
        else:
            raise RuntimeError("{0} is not a valid directory!".format(directory))
    else:
        raise RuntimeError("{0} is not a valid directory!".format(directory))


def setup_and_clear_segmentation_dir(segmentation_directory):
    check_and_create_dir(segmentation_directory)  # Checks if directory exists
    clear_directory(segmentation_directory)  # Asks for clearing the directory if not empty