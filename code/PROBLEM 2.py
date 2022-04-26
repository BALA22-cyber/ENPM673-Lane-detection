#### PROBLEM 2
import cv2 as cv
import numpy as np
import math
from utils2 import *
#### Change video directory
video = cv.VideoCapture("/home/kb2205/Desktop/ENPM 673/PROJECT 2/balase22_project2/input/whiteline.mp4")

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('Lane Detection.avi', fourcc, 30.0, (960,540))
while True:
    isTrue,frame = video.read()
    if not isTrue: break
    gray = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
    poly = gray.copy()
    polygon = np.array( [[[0,540],[100,540],[450,315],[530,315],[900,539],[959,540],[959,0],[0,0]]],dtype=np.int32)
    cv.fillPoly( poly , polygon, (0,0,0) )
    blur =  cv.GaussianBlur(poly,(5,5),cv.BORDER_DEFAULT)
    edge =  cv.Canny(blur,150,200)
    retval,threshed = cv.threshold(blur,150,250,cv.THRESH_BINARY)
    # cv.imshow("input",frame)
    cv.imshow("blur",blur)
    cv.imshow("canny",edge)
    cv.imshow("threshed",threshed)
    
    ### Line parameters
    rho = 1
    theta = (np.pi/180) * 1
    threshold = 15
    min_line_length = 1
    max_line_gap = 10
    
    hough_lines = hough_transform(threshed, rho, theta, threshold, min_line_length, max_line_gap)    
    separated_lanes = separate_lines(hough_lines, frame)
    img_different_lane_colors = color_lanes(frame, separated_lanes[0], separated_lanes[1])
    lane_colors =  np.uint8(img_different_lane_colors)
    cv.imshow("Separated Lanes ",lane_colors)
    out.write(lane_colors)
 
    if cv.waitKey(30) & 0xFF==ord("k"):
        break
video.release()
out.release()
cv.destroyAllWindows()