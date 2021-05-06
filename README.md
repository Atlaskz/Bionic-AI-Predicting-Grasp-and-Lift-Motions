## Realtime Prediction of Muscle Movement Using EEG-based Brain-Machine Interface

## Intoduction

This project aims to predict a person's intended hand and arm movement using their brain activity. The data was collected during an experiment where subjects performed a series of grasp and lift trials while wearing a 64 channel EEG headset. 

## Dataset

The [Dataset](https://www.kaggle.com/c/grasp-and-lift-eeg-detection) was uploaded to Kaggle and used for a competition in 2015. It contains data collected from 12 subjects, each performing 10 series of trials. [This video](https://youtu.be/XmgohaEAdjg) represents the 6 movements included within a single trial. The subjects were asked to perform the tasks as the light goes on. The graph below represents these tasks as a function of time. Each task is represented by 1 during its occurence and 0 otherwise.

![alt text](https://github.com/Atlaskz/Bionic-AI-Predicting-Grasp-and-Lift-Motions/blob/main/Images/motions.png?style=centerme)

The below figure represents information about the first task. The top figure represents the 32 electrodes that the signals are being collected from (in black). The bottom graph represents a downsampled version of the electrodes’ activity in a timeframe of 50 millisecond before, and 100 millisecond after the event onset (here Called “HandStart”). The sampling rate is 500 Hz, meaning the readings were collected every 2 milliseconds. 

![alt text](https://github.com/Atlaskz/Bionic-AI-Predicting-Grasp-and-Lift-Motions/blob/main/Images/e1.png?style=centerme)


The above graph has a distinct peak reflecting the action potential collected by electrodes placed on the motor cortex. However, such peaks are not always observable when using non invasive EEG devices. This is mainly due to the noise and artifacts resulting from the large distance between the electrodes and the signal source. The following are collected from task 2 and 3 of the trial:


![alt text](https://github.com/Atlaskz/Bionic-AI-Predicting-Grasp-and-Lift-Motions/blob/main/Images/e2.png?style=centerme)

![alt text](https://github.com/Atlaskz/Bionic-AI-Predicting-Grasp-and-Lift-Motions/blob/main/Images/e3.png?style=centerme)

In order to overcome this probelm, noise reduction methods such as Independent Component Analysis are used. However, this wasn't the main focus of this project.

## Training

Due to the timeseries nature of EEG data, previous timepoints are used as features during training. The input to the model is a 3d array of data in batches of size 2000. Each batch includes a single reading and 511 previous timepoints (window size = 512). However, to downsample the data, every other time point was used, resulting in a window size of 256. Given that data was colected from 32 electrodes, the shape of the arrays used in training were 2000x256x32.

There are 6 depended variables each representing one of the 6 tasks. When training the model, a binary cross entropy loss was used rather than categorical to allow for independent prediction of each task. This is because events overlapped at some instances of a each trial (see the figure above).

In order to compare performance, 3 scenarios were designed: 

1. Training the model for subject individually
2. Training the model on all subjects 
3. Training the Model once on all subjects and a second time on each subject individually

The aim of creating these scenarios was to explore the potential of creating a general model that performs well on new subjects.

For each scenario, a base model (Logistic Regression) and a Deep Learning Model (1D Convolutional Neural Network) were trained.

## Results

Below is a graph of the AUC score from the 6 models. AUC was used as the performance metric due to is its ability to measure separability of the 2 classes (0s and 1s), which is important for imbalanced data.

<p align="center">
  <img src="https://github.com/Atlaskz/Bionic-AI-Predicting-Grasp-and-Lift-Motions/blob/main/Images/results.png">
</p>

As shown in the figure above, scenario 1 performed the worst while the advanced model from scenario 3 performed best with an AUC score of 92.8%. This suggests that. It is also clear that the CNN performed better than the Log Reg model in all 3 scenarios. However, as you can see, score difference was small in all scenarios and the CNN took considerably longer than the base model to train. I would like to explore the possibility of improving the performance of the Logistic regression model without having to do any feature engineering. More on that below.

I have created a [short demo](https://youtu.be/HbB8mPIOpm0) showing the true events and the corresponding predictions that I got from the best model above. The lower graph represents the activity of the 32 electrodes during the trial. In the top graph, the true events are shown in orange and their prediction is shown right below each in blue. 

 you can see that the orange and blue lines align pretty nicely. The least overlap is seen in event 5, which could either be due to the event itself or the loss function I have defined for finding the threshold. This is another area I would like to explore. It is also worth mentioning that each event happened in a timeframe of 75 milliseconds, hence, in event 5, the prediction is delayed by less than 0.15 seconds, which doesn’t seem too bad for a first try.


## Wrapping up

I had a blast working on this problem as my first real data science project. If I decided to improve the accuracy, I would spend more time on signal processing to reduce the noise in the data. I would also try other deep learning algorithms and compare more models. I have a slight bias towards using deep learning for this problem due to its end to end optimization and the elimination of the feature engineering step. This is important since having to feature engineer the incoming data could increase prediction delay in real time. 

Speaking of real time, I thought it would be very interesting to test the performance of the model for actual prediction. Hence, for my next project, I am hoping to use a muse EEG headset or a similar product for predicting sentiment from brain activity in real time.

