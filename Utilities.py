#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:56:59 2019

@author: Thomas TOPILKO
"""

import os;

import TopKinectMouseTracker.Settings as settings;
import TopKinectMouseTracker.Params as params;


def PrintColoredMessage(msg,color) :
    
    '''Prints a colored message in the console
    
    Params : 
        msg (str) : the message to be printed
        color (str) : the color to be used for the message (see format in params.colors)
        
    Returns :
        prints the colored message in the console
    '''
    
    if isinstance(msg, str) :
    
        if color in params.colors :
            
            return print(params.colors[color]+msg+params.colors["off"]);
        
        else :
            
            PrintColoredMessage("[WARNING] The color {0} is not available!".format(color),"darkred");
            
    else :
        
        PrintColoredMessage("[WARNING] The message {0} is not in the correct format!".format(msg),"darkred");
    
def PrintLoadingBar(percent) :
    
    '''Prints a loading bar in the console
       e.g (0%) = '..................................................'
           (11.5%) = '######............................................'
           (100%) = '##################################################'
    
    Params :
        percent (int/float) : the % of the ongoing operation (e.g : 11.5 = 11.5%; 100 = 100%)
        
    Returns :
        prints the loading bar in the console
    '''
    
    #Checks if percent argument is correct
    if percent >= 0 :
    
        pattern = ['.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.',\
                   '.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.',\
                   '.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.',\
                   '.','.'];
                   
        for n,i in enumerate(pattern) :
            if n < percent/2 :
                pattern[n] = '#';
            else :
                pattern[n] = '.';
                
        pattern = "".join(pattern);
        return pattern;
    
    else :
        
        PrintColoredMessage("[WARNING] A progress cannot be negative right?","darkred");

def HoursMinutesSeconds(time) :
    
    '''Function that takes a time variable (s) and converts it into (hh,mm,ss) format
    
    Params : 
        time (int) : the number of seconds of a time/event
        
    Returns : 
        returns the time in (hh,mm,ss) format
    '''
    
    #Checks if time argument is correct
    if time >= 0 :

        remainingMinutes = time%3600;
        remainingSeconds = remainingMinutes%60;
        hours = int(time/3600);
        minutes = int(remainingMinutes/60);
        seconds = int(remainingSeconds);
    
        return hours,minutes,seconds;
    
    else :
        
        PrintColoredMessage("[WARNING] Time cannot be negative right?","darkred");

def PlaySound(n,sound) :
    
    '''Function that plays n times a sound
    /!\ [WARNING] This function only works on OS X systems
    
    Params :
        n (int) : the number of times to play the sound
        sound (str) : the identity of the sound to be played (see format in params.sounds)
        
    Output :
        Plays the sound "sound", "n" times from system speakers
    '''
    
    #Checks if n argument is correct
    if n == 0 or n < 0 :
        n = 1;
    
    #Checks if sound argument is correct
    if sound in params.sounds :
        
        #Checks that the operating system is OS X
        if settings._platform == "darwin" :
        
            for i in range(n) :
                os.system('afplay /System/Library/Sounds/'+str(sound)+'.aiff');
        
        else :
            
            PrintColoredMessage("[WARNING] The functions utils.PlaySound requieres OS X operating system to work!","darkred");
            
    else :
        
        PrintColoredMessage("[WARNING] The sound parameter for utils.PlaySound is unavailable or not correct!","darkred");
        
def CheckDirectoryExists(directory) :
    
    if not os.path.exists(directory) :
        
        os.mkdir(directory);
        PrintColoredMessage("[INFO] {0} has been created".format(directory),"darkgreen");
        
    else :
        
        pass;
        
        
