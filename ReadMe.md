# Object Detection with Histograms of Oriented Gradients in C++

You need to install OpenCV for C++ on your favorite IDE to run this project

This is a project for object detection of several object which are fed into Random Forest
We didn't use the Random Forest class of OpenCV, but implemented it ourselves with multiple decision trees and then make a majority voting for it.
Steps are as follows: 

1.) calculate the HOG Features of our images in the database
2.) Train Multiple Decision Trees (with different training data from above) with the HOG stuff
3.) Sliding Window over the new image and feed that into our random forest
4.) Obtain predictions for each window in the image 
5.) Non Maximum Suppression of all the bounding boxes (only keep those with the highest probability)
6.) Draw the image with our predicted bounding boxes.


