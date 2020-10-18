# Self Driving Car
## Problem Definition
To build the minimal version of a self-driving car, where car predicts the steering angle itself using front camera. Here the car contains the front view camera. The model captures the video frames from the front view camera and predicts the steering angle the car should rotate.

## How to Run
Download the repository 
1) To train the model: run 'train.py --epochs <no> --batch_size <no>'
  It saves the trained model to output folder. It can be further fine-tuned or used for prediction
2) To test the model: run test.py

## Dataset
Refer this: https://github.com/SullyChen/Autopilot-TensorFlow

The dataset contains a total of 45406 images along with their steering angles stored in data.txt file. We train the model using 80% of the data, later 20% of the data is used for prediction.

## Objective
Our objective is to predict the correct steering angle from the given test image of the road. Here, our loss is Mean Squared Error(MSE). Our goal is to reduce the MSE error as low as possible.

## Prerequisites
You need to have installed following softwares and libraries in your machine before running this project.<br>
* **Python 3:** https://www.python.org/downloads/<br>

### Libraries:<br>
1. **Tensorflow:** It is a deep learning library.<br>
pip install tensorflow<br>
2. **OpenCV:** It is used for processing images.<br>
pip install opencv-python<br>
3. Other libraries such as pandas, matplotlib, numpy and scipy.<br>

## Authors
• Aditya
<br>

## Acknowledgments
• Applied AI Course 
