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
cd folder/to/clone-into/
git clone https://github.com/Tom-top/TopMouseTracker
```

If you don't have Git on your computer you can install it following this [link](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

Once TMT has been cloned, make sure to activate your virtual environment (if you use one), launch Spyder and open the _RunOffline.py_ in the TopMouseTracker folder located at folder/to/clone-into/

Once TMT has been cloned try to run all the imports in the begining of the _RunOffline.py_ file.
If there is no problem, you are all set, ready to track some stuff !

## Running the tests





## Authors

* **Thomas TOPILKO** - *Initial work* -

## License

N.A

