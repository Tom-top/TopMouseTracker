#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:56:59 2019

@author: Thomas TOPILKO
"""
import errno
import os
import sys

import shutil
import smtplib

import top_mouse_tracker.parameters as params


def print_colored_message(msg, color):
    """Prints a colored message in the console
    
    Params : 
        msg (str) : the message to be printed
        color (str) : the color to be used for the message (see format in params.colors)
        
    Returns :
        prints the colored message in the console
    """
    
    if isinstance(msg, str):
        if color in params.colors:
            print(params.colors[color]+msg+params.colors["off"])
        else:
            print_colored_message("[WARNING] The color {0} is not available!".format(color), "darkred")
    else:
        print_colored_message("[WARNING] The message {0} is not in the correct format!".format(msg), "darkred")


def print_loading_bar(percent):  # FIXME: replace by tqdm
    """Prints a loading bar in the console
       e.g (0%) = '....................................................................................................'
           (11.5%) = '######...........................................................................................'
           (100%) = '##################################################################################################'
    
    Params :
        percent (int/float) : the % of the ongoing operation (e.g : 11.5 = 11.5%; 100 = 100%)
        
    Returns :
        prints the loading bar in the console
    """
    
    # Checks if percent argument is correct
    if percent >= 0:
        pattern = 100 * ['.']
        for n, i in enumerate(pattern):
            if n < percent:
                pattern[n] = '#'
            else:
                pattern[n] = '.'
                
        pattern = "".join(pattern)
        return pattern
    
    else:
        print_colored_message("[WARNING] A progress cannot be negative right?", "darkred")


def hours_minutes_seconds(time):
    """
    Function that takes a time variable (s) and converts it into (hh,mm,ss) format
    
    Params : 
        time (int) : the number of seconds of a time/event
        
    Returns : 
        returns the time in (hh,mm,ss) format
    """
    # Checks if time argument is correct
    if time >= 0:
        remaining_minutes = time % 3600
        remaining_seconds = remaining_minutes % 60
        hours = int(time/3600)
        minutes = int(remaining_minutes/60)
        seconds = int(remaining_seconds)

        return hours, minutes, seconds
    else:
        print_colored_message("[WARNING] Time cannot be negative right?", "darkred")


def play_sound(n, sound):
    """Function that plays n times a sound
    /!\ [WARNING] This function only works on OS X systems
    
    Params :
        n (int) : the number of times to play the sound
        sound (str) : the identity of the sound to be played (see format in params.sounds)
        
    Output :
        Plays the sound "sound", "n" times from system speakers
    """
    # Checks if n argument is correct
    if n == 0 or n < 0:
        n = 1
    
    # Checks if sound argument is correct
    if sound in params.sounds or sound is not None:  # TODO: check
        # Checks that the operating system is OS X
        if sys.platform == "darwin":
            for i in range(n):
                os.system('afplay /System/Library/Sounds/'+str(sound)+'.aiff')
        else:
            print_colored_message("[WARNING] The functions utils.PlaySound requires OS X operating system to work!",
                                  "darkred")
    else:
        if sys.platform == "linux":
            for i in range(n):
                os.system("/usr/bin/canberra-gtk-play --id='bell'")
        else:
            print_colored_message("[WARNING] The sound parameter for utils.PlaySound is unavailable or not correct!",
                                  "darkred")


def check_directory_exists(directory):
    ''' Function that checks is a certain directory exists. If not : creates it
    
    Params :
        directory (str) : path to the directory to be created
        
    Output : 
        Creates the designed directory 
    '''
    
    if isinstance(directory, str):
        if os.path.dirname(directory):
            if not os.path.exists(directory):
                os.mkdir(directory)
                print_colored_message("[INFO] {0} has been created".format(directory), "darkgreen")
            else:
                pass
        else:
            raise RuntimeError("{0} is not a valid directory!".format(directory))
    else:
        raise RuntimeError("{0} is not a valid directory!".format(directory))


def clear_directory(directory):
    if isinstance(directory, str):
        if os.path.dirname(directory):
            if os.listdir(directory):
                _input = input("Type [y]/n to clean {0} :   ".format(directory))
                if _input == 'y':
                    for obj in os.listdir(directory):
                        try:
                            os.remove(os.path.join(directory, obj))
                        except OSError as err:
                            if err.errno != errno.ENOENT:
                                raise
                            else:
                                shutil.rmtree(os.path.join(directory, obj))

                elif _input == 'n':
                    print_colored_message("{0} was kept untouched!".format(directory), "darkgreen")
                else:
                    print_colored_message("Wrong input!".format(directory), "darkgreen")
            else:
                print_colored_message("{0} was already empty!".format(directory), "darkgreen")
        else:
            raise RuntimeError("{0} is not a valid directory!".format(directory))
    else:
        raise RuntimeError("{0} is not a valid directory!".format(directory))


def check_mail():
    if params.main["email"] is not None:
        params.main["password"] = input("Type the password for your email {0} : "
                                        .format(params.main["email"]))
        try:
            s = smtplib.SMTP(params.main["smtp"], params.main["port"])
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(params.main["email"], params.main["password"])
            print_colored_message("Emailing mode has been enabled", "darkgreen")
        except smtplib.SMTPException as err:
            print_colored_message("[WARNING] Wrong Username or Password !", "darkred")
            print(err)
    else:
        print_colored_message("Emailing mode has been disabled", "darkgreen")


def check_directories():
    check_directory_exists(params.main["tmtDir"])  # Checks if directory exists
    check_directory_exists(params.main["resultDir"])  # Checks if directory exists
    clear_directory(params.main["resultDir"])  # Asks for clearing the directory if not empty
    
    if params.saving["fourcc"] == "Frames":
        check_directory_exists(params.main["segmentationDir"])  # Check if directory exists
        
    check_mail()
