[![build passing](https://travis-ci.org/ukubuka/ukubuka-core.svg?branch=master	)](https://github.com/AswinVasudevan21/HandGestureClassification/blob/master/README.md)
[![python version](	https://img.shields.io/badge/Python-3.6-blue.svg)](https://github.com/AswinVasudevan21/HandGestureClassification/blob/master/README.md)
[![tensorflow version](	https://img.shields.io/badge/Tensorflow-1.2-yellow.svg)](https://github.com/AswinVasudevan21/HandGestureClassification/blob/master/README.md)
[![keras version](	https://img.shields.io/badge/Keras-2.6-green.svg)](https://github.com/AswinVasudevan21/HandGestureClassification/blob/master/README.md)




# HandGestureClassification
### Objective 
The goal of this project is to classify the hand gestures using a dataset comprising of depth images obtained from a Kinect v2 Camera. The accuracy of the stored model is 99.14%

### Dataset
https://www.kaggle.com/gti-upm/depthgestrecog

### Steps in Execution:
On completion of the four steps below : preprocessing, training, prediction
and results will be displayed.
  
    1. Clone or Download the project
    2. Pip install the requirements.txt
    3. Run ModelPipeline.py on python 3
    4. Provide the local path to the dataset folder. 

### Architecture:
The classification problem is dealt using CNN. The compelling reason is
that the images have strong features with no background so am confident
that the nets can learn to distinguish the gestures and generalize. 

### Results:
Dataset split = 80% for training 10% each for validation and test
    
    1. Training:
      a. Val_loss =0.0380
      b. Val_accuracy = 0.9901
    2. Test:
      a. Loss = 0.0462
      b. Accuracy = 0.989

### Visualization:

#### Original Image and Augmented Image
<img height="300px" src="https://github.com/AswinVasudevan21/HandGestureClassification/blob/master/graphs/augmented.png">

#### Loss Train and Test
<img height="300px" src="https://github.com/AswinVasudevan21/HandGestureClassification/blob/master/graphs/loss.png">

#### Accuracy Train and Test
<img height="300px" src="https://github.com/AswinVasudevan21/HandGestureClassification/blob/master/graphs/accuracy.png">

### Future Work:
Experiment with hyper parameters and add dropout. 
