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

  I suggest running TMT in an integrated development environement (IDE). I personnaly use [PyCharm](https://www.jetbrains.com/pycharm/).

  ```
  conda install numpy
  conda install matplotlib
  conda install pandas
  conda install natsort
  conda install -c anaconda configobj
  conda install -c anaconda scipy
  conda install -c anaconda psutil
  conda install -c conda-forge pywin32-ctypes
  conda install -c conda-forge pyautogui
  conda install -c conda-forge opencv
  conda install -c conda-forge moviepy
  conda install -c conda-forge sk-video
  ```

### Installing

You can install TMT by cloning it with Git :

```
cd folder/to/clone-into
git clone https://github.com/Tom-top/TopMouseTracker
```

If you don't have Git on your computer you can install it following this [link](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

Once TMT has been cloned, make sure to activate your virtual environment (if you use one), and launch your favorite IDE.

To perform video recordings using the Kinect V2 open the run_kinect.py file.
For the tracker open the run_segmentation.py file.
Try to run all the imports and if there are no problems, you are all set, ready to record and track some nesting behavior!

## Running the tests

* You can download an example dataset [here](https://www.dropbox.com/sh/dfq8xtqkkynb2fl/AAC5TwxBchaXqNAjI_hK3tsDa?dl=0)

* You can then run the whole first section which will set all the TMT parameters used to run the test on the small data set   located at : folder/to/clone-into/TopMouseTracker/Test

* Next, you can run the second section of the script which will load the test movies into memory.

* The third section initializes the Tracker class

* Running the fourth section will allow you to set the region of interest (ROI) within the first/whatever frame of the video you are about to analyze. To do so : left-click with the mouse in the top-left corner of the ROI you want to trace and drag the mouse cursor until you reach the bottom-right corner of the desired ROI and release. This should set a ROI in the form of a red rectangle that represents the ROI within which the tracker will run. If you are ok with the ROI you just set, press "C", otherwise press "R" and restart over.

* /!\ OPTIONAL The fourth section of the script will allow you to fine tune the thresholding parameters for the segmentation.

* The fifth section of the code finally launches the tracker.

* For the analysis, the sixth section will allow you to generate tracking plots such as this one :

<a href="url"><img src="https://github.com/Tom-top/TopMouseTracker/blob/master/Images/TopMouseTracker_Tracking_Plot.png" align="center" ></a>

Giving you the track of the segmented animal across time with summary plots on the right.

* The seventh section finally allows you to synchronize the tracking of the animal with the tracking plot in live for display purposes mostly. An example :

![](https://github.com/Tom-top/TopMouseTracker/blob/master/Images/TopMouseTracker_Live_Tracking.gif)

## Authors

* **Thomas TOPILKO**

## License

N.A


