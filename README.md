
[//]: # (Image References)

[image1]: ./img/system.png "Generic Description 1" 
[image2]: ./img/wp.png "Generic Description 2" 
[image3]: ./img/dbw.png "Generic Description 3" 
[image4]: ./img/tl.png "Generic Description 4" 

# Capstone Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This repository contains all the code for the final project of Udacity Self Driving Car Nanodegree.

## Overview

In this project the goal is to write ROS nodes to implement core functionality of the autonomous vehicle system, including traffic light detection, control, and waypoint following! 

The code is developed in simulator and then tested on real self-driving car 


## Team members

Name | Github Account | Email
--- | --- | ---
Danilo Romano (Team Lead) | https://github.com/danyz91 | d.romano991@gmail.com
Gianluca Mastrorillo | https://github.com/Giangy1990 | mastrorillo.gianluca@gmail.com
Hossein Gholizadeh  | https://github.com/fpgholizadeh | fpgholizadeh@gmail.com
Sardhendu Mishra | https://github.com/Sardhendu | sardhendumishra@gmail.com

## Setup

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the three installation options.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Virtual machine
* Minimum configuration:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

To set up the environment, perform the following steps:
1. Download the [Udacity VM image](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Udacity_VM_Base_V1.0.0.zip). N.B. the password for the *student* user is **udacity-nd**
2. Import it using VirtualBox
3. Download the [simulator](https://github.com/udacity/CarND-Capstone/releases) for your host OS
4. open the port for communication between simulator and ros following the [guide](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/Port+Forwarding.pdf)

On the Ubuntu VM
1. Install virtualenv
```bash
sudo apt-get install python3-pip
sudo pip3 install virtualenv
```
2. Create a virtualenv that uses python2.7
```bash
virtualenv -p /usr/bin/python2.7 venv
```
3. Clone our repo
```bash
git clone https://github.com/danyz91/CarND-Capstone.git
```
4. Startup the virtualenv
```bash
source venv/bin/activate
```
5. Downgrade the importlib-resources
```bash
pip install importlib-resources==1.0
```
6. Install dependencies
```bash
pip install -r requirements.txt
```
7. Add the following line to the .bashrc
```bash
export PYTHONPATH=$PYTHONPATH:/usr/lib/python2.7/dist-packages
source /home/student/CarND-Capstone/ros/devel/setup.sh
```

From now on, you can use the virtualenv to run the CarND-Capstone code

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

## Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

## Real world testing

1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information

Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |


## System Overview

![alt text][image1]

## ROS Architecture

## Node Design

### Waypoint Updater

![alt text][image2]

### Control

![alt text][image3]

### Traffic Recognition

![alt text][image4]

#### Dataset Annotation

##### Installation

1. Labeling:
   * git clone https://github.com/tzutalin/labelImg.git
   * cd labelImg
   * conda create -n py3 python=3.5
   * conda activate py3
   * pip install pyqt5==5.13.2 lxml
   * make qt5py3
   * python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]


2. Modeline:
   * scikit-image
   *


#### References:

- Labeling: https://github.com/tzutalin/labelImg
- Voc to Coco converter: https://github.com/roboflow-ai/voc2coco
- Custom Data to Coco: https://github.com/waspinator/pycococreator
- Modeling: https://github.com/fizyr/keras-retinanet

