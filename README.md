# Lane detection practice
This project is praticing how to use histogram equalizations and detect straight and curve lane lines.
The prerequirement is to place adaptive_hist_data, whiteline.mp4, and challenge.mp4 under the file.
The environment is based on python3, OpenCV, and numpy.
## Structure
├── adaptive_hist_data  
|-- problem1.avi
├── challenge.mp4  
├── whiteline.mp4  
├── output1  
├── output2  
├── output3  
├── Problem1.py  
├── Problem2.py  
├── Problem3.py  
└── filters.py  
└──utils2.py

## Histogram equalizations
To run the code, please change the file directory and run
I've converted the video frames into a single video and saved it as " problem1.avi"

It will create two videos "histogramEqualization.avi" and "AdaptiveHistEqualization.avi".  
HistogramEqualization.avi will show the video after histogram equalizations  
AdaptiveHistogramEqualization.avi will show the video after adaptive histogram equalizations  
Original image (video)
![](output1/problem1.avi) 
Histogram equalized Video
![](output1/HistogramEqualization.avi) 
Adaptive histogram equalized Video
![](output1/AdaptiveHistogramEqualization.avi) 
## Straight Lane Detection  
To run the code, please change video Directory

It will create one video "lanedetection.avi" and show green for solid line and red for dashed line.  

## Predict Turn  
To run the code, please change video directories and make sure the video and filter.py are in the same directory
It will create one video "predict Turn.avi" and predicts the turn with color super Imposition
Take an video for example:  
Original video  
![](output3/test1.avi)   
Result video  
![](output3/result.avi) 
