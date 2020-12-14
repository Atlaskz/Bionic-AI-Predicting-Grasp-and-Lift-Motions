# Bionic AI

## Intoduction

Around the same time I started on my data science journey, I took an interest in the functionality of the human brain. This is when I began doing more research on Brain Computer Interfaces (BCI). These are devices that allow a direct communication pathway between the brain and an external device. Our thoughts are transmitted through our brain as a series of electrical impulses. These are then picked up by a computer from the brain using EEG(electroencephalography) and from muscles using EMG(electromyography). 

BCIs are not a new technology, they started being used in the 70s for understanding the brain. However, in the past few years, researchers have been looking at whether these electrical signals can be decoded to give an insight into the person’s thoughts, or in other words, read their mind.

## Applications

How does this create an impact? Well, being able to analyze a person’s thoughts means the opportunity to predict their mental and emotional state as well as their muscle movements. One of the significant applications of muscle movement prediction is in creating robotic prosthetics. There are currently companies that develop prosthetics that use the amputee’s brain activity for movement and control, expanding their range of functionality to a new level.

Applications of thought predictions include helping individuals with paralysis control wheelchairs, enabling individuals with speech disorders communicate their thoughts, and even allowing people speaking different languages communicate without speaking.

Eventually, we might even be able to communicate with our beloved pets!


## Using EEG Signals for Predicting Hand and Arm Movements

For my first data science project, I decided to explore EEG data because of my interest in the human brain. I was stuck between predicting motions and thoughts. However, it turns out that predicting thoughts is a much broader and more abstract problem. Hence, I went with predicting motions. In this post I’ll go over my project, which is using people’s brain activity to predict their hand/arm movements. 

## Dataset

The Dataset I used is from a Kaggle Competition (LINK) that includes EEG data from subjects performing a series of grasp and lift trials. There were 12 subjects that took part in the experiment, each performing 10 series of trials, and each series included around 30 trials. The video on the right hand side represents a single trial. The subject was asked to perform the tasks as soon as the light goes on. The graph right below the video represents the 6 tasks during a single trial. Each task is given a value of 1 if it’s happening at a point in time and 0 otherwise.
 
Let’s take a closer look at a single event within a trial. The top figure represents the 32 electrodes that the signals are being collected from (in black). The bottom graph shows a smoothed version of these electrodes’ activity in a timeframe of 50 millisecond before, and 100 millisecond after the event onset (Called “HandStart”). Notice that the sampling rate is 500 Hz, which means the readings were collected every 2 milliseconds. 

You can see how the subject’s brain activity changes as they perform the task. It is worth mentioning that seeing this defined pattern is not always so easy. Usually the signals are affected by noise and artifacts from the surroundings and the person’s thoughts. Have a look at the other 5 events from the same subject:






Because of this, the task of identifying patterns and making predictions based on them is not always so easy, this is why as many as 30 trials were performed for each subject.

## Training

Since this is a timeseries analysis, it is very important to use previous timepoints for future predictions. The input to my ML algorithm is batches of data, each with a size of 2000. Each of the 2000 sample points also includes 512 prior timepoints (window size = 512). However, I considered every second point in order to reduce the training time. Hence, my batch size was 2000x256x32.

On the other side, I have 6 outputs for the 6 different events. I treated each event as a binary classification problem rather than categorical classification since more than one event can happen at once.

In order to compare different models, I created 3 scenarios: 

Training the model on each subject’s EEG data individually
Training the model on all subjects EEG data
Training the Model once on all the EEG data and a second time on each subjects data individually

The reason behind using these scenarios was that I was curious if people’s brain activity is similar when they perform the same task, and if a general model can be used to predict motion for any individual.

For each scenario, I trained a base model (Logistic Regression) as well as Deep Learning Model (CNN) in order to compare the results from using deep learning vs traditional AI algorithms for decoding brain activity.

## Results

Below is a graph of the AUC score from my 6 models. AUC score was what was required by the actual competition. The reason for using AUC as a performance metric is its ability to measure separability of the 2 classes (0s and 1s), which is very important for imbalanced data such as ours.




To my surprise, scenario 1 performed the worst (I was expecting the general model from scenario 2 to be the worst). While the advanced model from scenario 3 performed best with an AUC score of 92.8%. This suggests that the model can successfully learn from patterns that are more defined in some subjects and use them towards making predictions for others. It is also clear that the CNN performed better than the Log Reg model in all 3 scenarios. However, as you can see, score difference was small in all scenarios and the CNN took considerably longer than the base model to train. I would like to explore the possibility of improving the performance of the Logistic regression model without having to do any feature engineering. More on that below.

I have created a short demo showing the true events and the corresponding predictions that I got from the best model above. The lower graph represents the activity of the 32 electrodes during the trial. In the top graph, the true events are shown in orange and their prediction is shown right below each in blue. 

 you can see that the orange and blue lines align pretty nicely. The least overlap is seen in event 5, which could either be due to the event itself or the loss function I have defined for finding the threshold. This is another area I would like to explore. It is also worth mentioning that each event happened in a timeframe of 75 milliseconds, hence, in event 5, the prediction is delayed by less than 0.15 seconds, which doesn’t seem too bad for a first try.




## Wrapping up

I had a blast working on this problem as my first real data science project. If I decided to improve the accuracy, I would spend more time on signal processing to reduce the noise in the data. I would also try other deep learning algorithms and compare more models. I have a slight bias towards using deep learning for this problem due to its end to end optimization and the elimination of the feature engineering step. This is important since having to feature engineer the incoming data could increase prediction delay in real time. 

Speaking of real time, I thought it would be very interesting to test the performance of the model for actual prediction. Hence, for my next project, I am hoping to use a muse EEG headset or a similar product for predicting sentiment from brain activity in real time.

