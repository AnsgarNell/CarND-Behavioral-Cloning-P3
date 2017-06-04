# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes. 

The model includes RELU layers to introduce nonlinearity (code line 67), and the data is normalized in the model using a Keras lambda layer (code line 66). 

#### 2. Attempts to reduce overfitting in the model

The model was tested with dropout layers in order to reduce overfitting (model.py lines 68), but it was removed due to the worse results obtained. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 29-46). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 72).

#### 4. Appropriate training data

Training data was taken from the download provided by Udacity, as some tests were made with own data sets that didn't achieve good results.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture in the end was to keep it simple, as adviced by forum mentors.

My first step was to use a convolution neural network model similar to the one used in the videos and LeNet as I thought this model might be appropriate because it was used before.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The results were very bad when using data gathered by myself. So a test was made with the data available in the web, and the vehicle behavior increased a lot.

Then I look for some advice in the forums, and as said before, decided to try a simple model approach. This gave better results so this was the startegy to follow.

Using generators was a very good step to make the training process less resource consuming, as I have no GPU that can be used with TensorFlow. A key step was to reduce the batch size to a value from 32.

After that, and reading that having a lot of data with 0º steering could bias the model to this angle, approximately 3000 samples were deleted containing this angle value. This also resulted in a performance increase.

The next setp was to use the images from all three angles, and after this no much improvements were achieved despite of trying a dropout layer (which was deleted afterwards, using flipped images and measurements (still implemented in the final code) and testing different values for the number of epochs variable. This began with a value of 20, which was way too much as no improvements in the loss were obtained after epoch number 4 or 5 (final value).

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle sometimes fell off the track, like after the bridge when the right line disappears and the alternative path begins. To improve the driving behavior in these cases, I recorded additional data for this places. However, the car sometimes goes right into the path or crashes against the wall.  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road, as can be seen in the video run1.mp4.

#### 2. Final Model Architecture

My model consists of a convolution neural network with 5x5 filter sizes. 

The model includes RELU layers to introduce nonlinearity (code line 67), and the data is normalized in the model using a Keras lambda layer (code line 66). 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving, one clockwise and other in the oposite direction, but due to the bad results decided to use the downloaded data.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
