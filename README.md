[![DOI](https://zenodo.org/badge/166059052.svg)](https://zenodo.org/badge/latestdoi/166059052)

# TopMouseTracker

<a href="url"><img src="https://github.com/Tom-top/TopMouseTracker/blob/master/Images/TopMouseTracker_Logo.png" align="center" height="523" width="781" ></a>


TopMouseTacker is a Python library based on OpenCV made for object tracking in videos.

## Getting Started

### Prerequisites

The easiest way to get TopMouseTracker (TMT) running is to install [Anaconda](https://www.anaconda.com/distribution/).

To install all the pre-requiered libraries for TMT there are two main options :

* Cloning the TMT environment using a .yml file :

  For this, install TMT first by following the instructions under "Installing" and then run :

  ```
  conda env create --n envname -f path/to/the/TMT_Environment.yml
  ```

* Installing all the pre-requiered libraries manually :

  I suggest running TMT in an integrated development environement. I personnaly use [Spyder](https://www.anaconda.com/distribution/). You can install Spyder and all the other pre-requiered libraries using Anaconda very easily by executing the following commands :

  ```
  conda install spyder
  conda install numpy
  conda install matplotlib
  conda install pandas
  conda install xlwt
  conda install xlrd
  conda install natsort
  conda install pyqt=5.9.2=py36h05f1152_2
  conda install -c menpo opencv3=3.1.0
  conda install -c conda-forge imageio-ffmpeg
  conda install -c conda-forge scikit-video
  conda install -c conda-forge moviepy
  ```

### Installing

You can install TMT by cloning it with Git :

```
cd folder/to/clone-into
git clone https://github.com/Tom-top/TopMouseTracker
```

If you don't have Git on your computer you can install it following this [link](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

Once TMT has been cloned, make sure to activate your virtual environment (if you use one), launch Spyder and open the _RunOffline.py_ in the TopMouseTracker folder located at folder/to/clone-into/

Once TMT has been cloned try to run all the imports in the begining of the _RunOffline.py_ file.
If there is no problem, you are all set, ready to track some stuff !

## Running the tests

First change the _topMouseTrackerDir_ parameters and set it to where you cloned TMT :

```
_topMouseTrackerDir = folder/to/clone-into/TopMouseTracker
```

* You can then run the whole first section which will set all the TMT parameters used to run the test on the small data set   located at : folder/to/clone-into/TopMouseTracker/Test

* Next, you can run the section section of the script which will load the test movies into memory and will initialize the Tracker class.

* Running the third section will allow you to set the region of interest (ROI) within the first/whatever frame of the video you are about to analyze. To do so : left-click with the mouse in the top-left corner of the ROI you want to trace and drag the mouse cursor until you reach the bottom-right corner of the desired ROI and release. This should set a ROI in the form of a red rectangle that represents the ROI within which the tracker will run. If you are ok with the ROI you just set, press "C", otherwise press "R" and restart over.

* /!\ OPTIONAL The fourth section of the script will allow you to fine tune the thresholding parameters for the segmentation.

* The fifth section of the code finally lauches the tracker.

* For the analysis, the sixth section will allow you to generate tracking plots such as this one :

<a href="url"><img src="https://github.com/Tom-top/TopMouseTracker/blob/master/Images/TopMouseTracker_Tracking_Plot.png" align="center" ></a>

Giving you the track of the tracked object in time with some summary plots on the right.

* The seventh section generates the summary plots for nest-building segmentation only.

* The eight section finally allows you to synchronize the tracking of the animal with the tracking plot in live for display purposes mostly. An example :

![](https://github.com/Tom-top/TopMouseTracker/blob/master/Images/TopMouseTracker_Live_Tracking.gif)

## Authors

* **Thomas TOPILKO**

## License

N.A


